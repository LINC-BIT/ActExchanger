import argparse
import os
import sys
from datetime import datetime

os.environ.setdefault("MS_ASSET_DIR", os.path.expanduser("~/.maniskill"))
sys.path.append(os.getcwd())

from train.toy_cnn.multi_agents.two_robot_pick.gpu_auto_select import configure_cuda_visible_devices

configure_cuda_visible_devices()

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.io_utils import dump_json, load_json
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train.marl.mappo.base import collect_rollout, mappo_update_on_policy
from train.reinforcement_learning.evaluate import evaluate
from train.reinforcement_learning.make_env import make_eval_envs
from train.reinforcement_learning.utils import compute_gae
from train.toy_cnn.multi_agents.place_cucumber.model import HeteroMAPPOAgent
import envs.pass_object_five_robots  # noqa: F401


MODEL_NAME = "toy_cnn_pass_object_mappo_pretrain"
CAMERAS = ("base_camera",)


class FlattenRGBObservationWrapperForMARL(gym.ObservationWrapper):
    def __init__(self, env, agent_obs_rules, rgb=True, state=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_state = state
        self.agent_obs_rules = agent_obs_rules
        self.state_clip = 10.0

        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "rgb" not in first_cam:
            self.include_rgb = False
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def _sanitize_state_tensor(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=self.state_clip, neginf=-self.state_clip)
        return torch.clamp(x, min=-self.state_clip, max=self.state_clip)

    def observation(self, observation: dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        rgb_images = {}
        for k, cam_data in sensor_data.items():
            if self.include_rgb:
                rgb_images[k] = cam_data["rgb"]

        agent_states = {}
        for agent_name in observation["agent"].keys():
            selected = {}
            for key in self.agent_obs_rules[agent_name]:
                if key in observation["agent"][agent_name]:
                    selected[key] = observation["agent"][agent_name][key].to(self.base_env.device)
                elif key in observation["extra"]:
                    selected[key] = observation["extra"][key].to(self.base_env.device)
                else:
                    raise KeyError(f"{key} not found for {agent_name}")
            agent_states[agent_name] = self._sanitize_state_tensor(
                common.flatten_state_dict(
                    selected, use_torch=True, device=self.base_env.device
                )
            )

        global_state = self._sanitize_state_tensor(
            common.flatten_state_dict(
                observation, use_torch=True, device=self.base_env.device
            )
        )

        ret = {}
        if self.include_rgb:
            ret["rgb"] = rgb_images["base_camera"]
        if self.include_state:
            for name, state in agent_states.items():
                ret[f"agent_states_{name}"] = state
            ret["global_state"] = global_state
        return ret


def resolve_ckpt_dir(args):
    task_dir = os.path.join(
        args.save_dir,
        f"{args.task_name}/ppo/{args.robot_name}/{MODEL_NAME}",
    )
    root_dir = os.path.join(task_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(task_dir, exist_ok=True)
    resume_dir = args.resume_dir if args.resume_dir is not None else root_dir
    return {
        "task_dir": task_dir,
        "root_dir": root_dir,
        "log_dir": os.path.join(resume_dir, "tb"),
        "video_dir": os.path.join(resume_dir, "videos"),
        "latest_agent": os.path.join(resume_dir, "latest_agent.pt"),
        "best_agent": os.path.join(resume_dir, "best_agent.pt"),
        "latest_opt": os.path.join(resume_dir, "latest_opt.pt"),
        "metrics": os.path.join(resume_dir, "metrics.json"),
    }


def normalize_stage_mode(stage_mode):
    if stage_mode is None:
        return "full"
    if isinstance(stage_mode, str):
        mode = stage_mode.strip().lower()
        if mode in ("", "full", "all"):
            return "full"
    else:
        mode = str(int(stage_mode))
    stage_id = int(mode)
    if stage_id < 0 or stage_id > 4:
        raise ValueError(f"stage_mode must be one of 0,1,2,3,4,full, got {stage_mode!r}")
    return str(stage_id)


def parse_curriculum_plan(args):
    if args.curriculum_stages:
        stages = [
            normalize_stage_mode(token)
            for token in args.curriculum_stages.split(",")
            if token.strip()
        ]
        if not stages:
            raise ValueError("curriculum_stages is empty after parsing")
        if args.curriculum_stage_steps:
            steps = [
                int(token.strip())
                for token in args.curriculum_stage_steps.split(",")
                if token.strip()
            ]
            if len(steps) != len(stages):
                raise ValueError(
                    "curriculum_stage_steps must have the same length as curriculum_stages"
                )
        else:
            base_steps = args.total_steps // len(stages)
            remainder = args.total_steps % len(stages)
            steps = [base_steps + (1 if i < remainder else 0) for i in range(len(stages))]
        return list(zip(stages, steps))
    return [(normalize_stage_mode(args.stage_mode), args.total_steps)]


def stage_tag(stage_mode):
    return "full" if stage_mode == "full" else f"stage_{stage_mode}"


def stage_video_tag(stage_mode):
    return "full_video_train" if stage_mode == "full" else f"stage_{stage_mode}_video_train"


def get_stage_active_agent_names(stage_mode, agent_names):
    if stage_mode == "full":
        return None
    stage_id = int(stage_mode)
    if stage_id < 0 or stage_id >= len(agent_names):
        raise ValueError(
            f"Invalid stage_mode={stage_mode!r} for agent_names={agent_names}"
        )
    return [agent_names[stage_id]]


def get_stage_agent_name(stage_mode, agent_names):
    active_names = get_stage_active_agent_names(stage_mode, agent_names)
    if active_names is None or len(active_names) != 1:
        raise ValueError(
            f"stage_mode={stage_mode!r} does not map to a single active agent"
        )
    return active_names[0]


def parse_stage_mode_list(stage_modes):
    if stage_modes is None:
        return []
    if isinstance(stage_modes, str):
        tokens = [token.strip() for token in stage_modes.split(",") if token.strip()]
    else:
        tokens = [str(stage_modes).strip()]
    return [normalize_stage_mode(token) for token in tokens]


def stage_max_episode_steps(stage_mode, args):
    if stage_mode == "full":
        return args.full_max_episode_steps
    if stage_mode != "0":
        return args.post_stage0_max_episode_steps
    return args.stage_max_episode_steps


def resolve_resume_from_stage(args, curriculum_plan):
    if not args.resume_from_stage:
        return None
    resume_stage = normalize_stage_mode(args.resume_from_stage)
    curriculum_stage_modes = [stage_mode for stage_mode, _ in curriculum_plan]
    if resume_stage not in curriculum_stage_modes:
        raise ValueError(
            f"resume_from_stage={resume_stage!r} is not in curriculum_plan={curriculum_stage_modes}"
        )
    return resume_stage


def stage_start_steps_from_plan(curriculum_plan):
    stage_start_steps = []
    cumulative_stage_steps = 0
    for _, planned_stage_steps in curriculum_plan:
        stage_start_steps.append(cumulative_stage_steps)
        cumulative_stage_steps += planned_stage_steps
    return stage_start_steps


def reset_agent_specific_actor_weights(agent, agent_name):
    reset_parts = []

    if getattr(agent, "actor_state_rms", None) is not None and agent_name in agent.actor_state_rms:
        rms = agent.actor_state_rms[agent_name]
        if hasattr(rms, "mean"):
            rms.mean.data.zero_()
        if hasattr(rms, "var"):
            rms.var.data.fill_(1.0)
        if hasattr(rms, "count"):
            rms.count.data.zero_()
        reset_parts.append("actor_state_rms")

    if agent_name in agent.actor_state_encoders:
        agent.actor_state_encoders[agent_name].apply(
            lambda module: module.reset_parameters()
            if hasattr(module, "reset_parameters")
            else None
        )
        reset_parts.append("actor_state_encoders")

    if agent_name in agent.actor_heads:
        agent.actor_heads[agent_name].apply(
            lambda module: module.reset_parameters()
            if hasattr(module, "reset_parameters")
            else None
        )
        reset_parts.append("actor_heads")

    if agent_name in agent.actor_logstd:
        with torch.no_grad():
            agent.actor_logstd[agent_name].fill_(-0.5)
        reset_parts.append("actor_logstd")

    return reset_parts


def reset_agent_specific_entropy(agent, agent_name, logstd_init: float = -0.5):
    reset_parts = []
    if agent_name in agent.actor_logstd:
        with torch.no_grad():
            agent.actor_logstd[agent_name].fill_(float(logstd_init))
        reset_parts.append("actor_logstd")
    return reset_parts


def extract_eval_success_score(eval_metrics):
    for key in ("success_rate", "success_once", "success_at_end"):
        if key in eval_metrics:
            return float(np.asarray(eval_metrics[key]).mean())
    first_key = next(iter(eval_metrics.keys()))
    return float(np.asarray(eval_metrics[first_key]).mean())


def extract_eval_success_episode_len_mean(eval_metrics):
    step_key = "success_first_step" if "success_first_step" in eval_metrics else "episode_len"
    if step_key not in eval_metrics:
        return float("nan")
    success_key = None
    for key in ("success_once", "success_at_end", "success_rate"):
        if key in eval_metrics:
            success_key = key
            break
    if success_key is None:
        return float("nan")
    success_mask = np.asarray(eval_metrics[success_key]) > 0.5
    if not bool(success_mask.any()):
        return float("nan")
    step_values = np.asarray(eval_metrics[step_key])
    return float(step_values[success_mask].mean())


def stage_gate_satisfied(score, success_episode_len_mean, success_threshold, success_episode_len_max):
    if score < success_threshold:
        return False
    if success_episode_len_max <= 0:
        return True
    return np.isfinite(success_episode_len_mean) and success_episode_len_mean <= success_episode_len_max


def make_env_kwargs(stage_mode, shader_dir, args, include_sim_backend=True):
    env_kwargs = {
        "obs_mode": "rgb+state_dict",
        "control_mode": "pd_joint_delta_pos",
        "render_mode": "rgb_array",
        "reward_mode": "normalized_dense",
        "shader_dir": shader_dir,
        "stage_mode": stage_mode,
        "auto_stage3_rotate_release": False,
        "max_episode_steps": stage_max_episode_steps(stage_mode, args),
    }
    if include_sim_backend:
        env_kwargs["sim_backend"] = "physx_cuda"
    return env_kwargs


def make_collate_fn(device):
    def _resize(img, size=128):
        return F.interpolate(img, size=size, mode="bilinear")

    def collate_fn(obs):
        if isinstance(obs["rgb"], np.ndarray):
            rgb = torch.from_numpy(obs["rgb"]).permute(0, 3, 1, 2).float() / 255.0
            global_state = torch.from_numpy(obs["global_state"])
            agent_states = {
                k: torch.from_numpy(v)
                for k, v in obs.items()
                if k not in ["rgb", "global_state"]
            }
        else:
            rgb = obs["rgb"].permute(0, 3, 1, 2).float() / 255.0
            global_state = obs["global_state"]
            agent_states = {
                k: v
                for k, v in obs.items()
                if k not in ["rgb", "global_state"]
            }

        rgb = _resize(rgb).to(device)
        global_state = global_state.to(device)
        batch = {"rgb": rgb, "global_state": global_state}
        for k, v in agent_states.items():
            batch[k] = v.to(device)
        return batch

    return collate_fn


def make_sample_fn(agent, collate_fn, deterministic=True, active_agent_names=None):
    active_agent_set = None if active_agent_names is None else set(active_agent_names)

    def _zero_like(action):
        if torch.is_tensor(action):
            return torch.zeros_like(action)
        return np.zeros_like(action)

    def sample_fn(obs):
        actions = agent.get_action(collate_fn(obs), deterministic=deterministic)
        if active_agent_set is not None:
            for name in list(actions.keys()):
                if name not in active_agent_set:
                    actions[name] = _zero_like(actions[name])
        return actions

    return sample_fn


def get_agent_info(args, env_kwargs, agent_obs_rules, collate_fn):
    env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend="gpu",
        env_kwargs=env_kwargs,
        wrappers=[lambda e: FlattenRGBObservationWrapperForMARL(e, agent_obs_rules=agent_obs_rules)],
    )
    obs, _ = env.reset(seed=args.seed)
    batch = collate_fn(obs)
    agent_names = list(agent_obs_rules.keys())
    state_dims = {
        name: int(batch[f"agent_states_{name}"].shape[-1])
        for name in agent_names
    }
    global_state_dim = int(batch["global_state"].shape[-1])
    action_dims = {
        name: int(env.unwrapped.agent.agents_dict[name].single_action_space.shape[0])
        for name in agent_names
    }
    env.close()
    return {
        "agent_names": agent_names,
        "state_dims": state_dims,
        "global_state_dim": global_state_dim,
        "action_dims": action_dims,
    }


def build_agent_obs_rules(agent_names):
    shared_extra = [
        "cube_pos",
        "cube_q",
        "panda_tcp",
        "so100_tcp",
        "widowx_tcp",
        "xarm6_tcp",
        "hand_palm",
        "handoff_target_01",
        "handoff_target_12",
        "handoff_target_23",
        "palm_target",
        "current_stage",
    ]
    tcp_to_cube_keys = {}
    for name in agent_names:
        if name.startswith("panda"):
            tcp_to_cube_keys[name] = "panda_tcp_to_cube_pos"
        elif name.startswith("so100"):
            tcp_to_cube_keys[name] = "so100_tcp_to_cube_pos"
        elif name.startswith("widowx"):
            tcp_to_cube_keys[name] = "widowx_tcp_to_cube_pos"
        elif name.startswith("xarm6"):
            tcp_to_cube_keys[name] = "xarm6_tcp_to_cube_pos"
        elif "inspire" in name or name.startswith("fixed_inspire_hand"):
            tcp_to_cube_keys[name] = "hand_palm_to_cube_pos"
        else:
            raise KeyError(f"Unsupported agent name for tcp_to_cube mapping: {name}")
    return {
        name: ["qpos", "qvel", tcp_to_cube_keys[name], *shared_extra]
        for name in agent_names
    }


def _state_dict_is_finite(state_dict):
    bad_keys = []
    for key, value in state_dict.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            bad_keys.append(key)
    return len(bad_keys) == 0, bad_keys


def _load_agent_state(agent, path, *, label):
    state = torch.load(path, map_location="cpu")
    finite, bad_keys = _state_dict_is_finite(state)
    if not finite:
        raise RuntimeError(
            f"{label} has non-finite tensors: path={path}, bad_keys={bad_keys[:8]}"
        )
    agent.load_state_dict(state)
    print(f"[MAPPO Pretrain] loaded {label}: {path}")
    return path


def _maybe_load_module_state(module, state_dict, prefix):
    sub_state = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            sub_state[key[len(prefix):]] = value
    if not sub_state:
        return False
    module.load_state_dict(sub_state)
    return True


def clear_agent_specific_optimizer_state(optimizer, agent, agent_name):
    reset_params = []
    if getattr(agent, "actor_state_encoders", None) is not None and agent_name in agent.actor_state_encoders:
        reset_params.extend(list(agent.actor_state_encoders[agent_name].parameters()))
    if getattr(agent, "actor_heads", None) is not None and agent_name in agent.actor_heads:
        reset_params.extend(list(agent.actor_heads[agent_name].parameters()))
    if getattr(agent, "actor_logstd", None) is not None and agent_name in agent.actor_logstd:
        reset_params.append(agent.actor_logstd[agent_name])
    for param in reset_params:
        optimizer.state.pop(param, None)


def load_single_stage_agent_best(agent, optimizer, ckpt, stage_mode, agent_names):
    stage_mode = normalize_stage_mode(stage_mode)
    if stage_mode == "full":
        raise ValueError("resume_load_agent_best_stage must be one of 0,1,2,3,4, not full")
    stage_best_tag = stage_tag(stage_mode)
    stage_best = os.path.join(
        os.path.dirname(ckpt["best_agent"]),
        f"best_agent_{stage_best_tag}.pt",
    )
    if not os.path.exists(stage_best):
        raise FileNotFoundError(
            f"Stage-best checkpoint not found for single-agent load mode: {stage_best}"
        )
    state = torch.load(stage_best, map_location="cpu")
    finite, bad_keys = _state_dict_is_finite(state)
    if not finite:
        raise RuntimeError(
            f"single_stage_agent_best({stage_best_tag}) has non-finite tensors: "
            f"path={stage_best}, bad_keys={bad_keys[:8]}"
        )
    agent_name = get_stage_agent_name(stage_mode, agent_names)
    loaded_parts = []
    if getattr(agent, "actor_state_rms", None) is not None and agent_name in agent.actor_state_rms:
        if _maybe_load_module_state(
            agent.actor_state_rms[agent_name],
            state,
            f"actor_state_rms.{agent_name}.",
        ):
            loaded_parts.append("actor_state_rms")
    if getattr(agent, "actor_state_encoders", None) is not None and agent_name in agent.actor_state_encoders:
        if _maybe_load_module_state(
            agent.actor_state_encoders[agent_name],
            state,
            f"actor_state_encoders.{agent_name}.",
        ):
            loaded_parts.append("actor_state_encoders")
    if getattr(agent, "actor_heads", None) is not None and agent_name in agent.actor_heads:
        if _maybe_load_module_state(
            agent.actor_heads[agent_name],
            state,
            f"actor_heads.{agent_name}.",
        ):
            loaded_parts.append("actor_heads")
    actor_logstd_key = f"actor_logstd.{agent_name}"
    if getattr(agent, "actor_logstd", None) is not None and agent_name in agent.actor_logstd:
        if actor_logstd_key in state:
            with torch.no_grad():
                agent.actor_logstd[agent_name].copy_(state[actor_logstd_key])
            loaded_parts.append("actor_logstd")
    clear_agent_specific_optimizer_state(optimizer, agent, agent_name)
    print(
        f"[Curriculum] loaded single-agent stage best: stage={stage_best_tag}, "
        f"agent={agent_name}, parts={loaded_parts}, path={stage_best}"
    )
    return stage_best, agent_name, loaded_parts


def load_multi_stage_agent_bests(agent, optimizer, ckpt, stage_modes, agent_names):
    parsed_stage_modes = parse_stage_mode_list(stage_modes)
    if not parsed_stage_modes:
        return []
    seen = set()
    ordered_stage_modes = []
    for stage_mode in parsed_stage_modes:
        if stage_mode in seen:
            continue
        seen.add(stage_mode)
        ordered_stage_modes.append(stage_mode)
    loaded = []
    for stage_mode in ordered_stage_modes:
        loaded.append(
            load_single_stage_agent_best(agent, optimizer, ckpt, stage_mode, agent_names)
        )
    return loaded


def load_resume_state(agent, optimizer, ckpt):
    loaded_agent_path = None
    loaded_latest = False
    resumed_step = 0
    if os.path.exists(ckpt["latest_agent"]):
        try:
            _load_agent_state(agent, ckpt["latest_agent"], label="latest_agent")
            loaded_agent_path = ckpt["latest_agent"]
            loaded_latest = True
        except RuntimeError as exc:
            print(
                "[MAPPO Pretrain] latest_agent has non-finite tensors; "
                f"fallback to best_agent. error={exc}"
            )
    if loaded_agent_path is None and os.path.exists(ckpt["best_agent"]):
        _load_agent_state(agent, ckpt["best_agent"], label="best_agent")
        loaded_agent_path = ckpt["best_agent"]
    if loaded_latest and os.path.exists(ckpt["latest_opt"]):
        latest_opt = torch.load(ckpt["latest_opt"], map_location="cpu")
        optimizer.load_state_dict(latest_opt["opt"])
        resumed_step = int(latest_opt.get("step", 0))
    if resumed_step > 0:
        print(f"[MAPPO Pretrain] resumed global_steps={resumed_step}")
    return resumed_step


def load_resume_from_previous_stage_best(agent, ckpt, curriculum_plan, resume_from_stage):
    resume_stage_index = next(
        idx for idx, (stage_mode, _) in enumerate(curriculum_plan)
        if stage_mode == resume_from_stage
    )
    if resume_stage_index <= 0:
        raise ValueError(
            "resume_reset_stage_agent requires RESUME_FROM_STAGE to have a previous stage "
            f"in curriculum_plan, got resume_from_stage={resume_from_stage!r}"
        )
    prev_stage_mode = curriculum_plan[resume_stage_index - 1][0]
    prev_stage_tag = stage_tag(prev_stage_mode)
    prev_stage_best = os.path.join(
        os.path.dirname(ckpt["best_agent"]),
        f"best_agent_{prev_stage_tag}.pt",
    )
    if not os.path.exists(prev_stage_best):
        raise FileNotFoundError(
            f"Previous-stage best checkpoint not found for resume reset mode: {prev_stage_best}"
        )
    _load_agent_state(agent, prev_stage_best, label=f"previous_stage_best({prev_stage_tag})")
    resumed_step = stage_start_steps_from_plan(curriculum_plan)[resume_stage_index]
    print(
        f"[Curriculum] resume reset mode: start {stage_tag(resume_from_stage)} from "
        f"{prev_stage_tag} best checkpoint, set global_steps={resumed_step}"
    )
    return resumed_step


def load_resume_from_stage_best(agent, ckpt, curriculum_plan, resume_from_stage):
    resume_stage_index = next(
        idx for idx, (stage_mode, _) in enumerate(curriculum_plan)
        if stage_mode == resume_from_stage
    )
    resume_stage_tag = stage_tag(resume_from_stage)
    stage_best = os.path.join(
        os.path.dirname(ckpt["best_agent"]),
        f"best_agent_{resume_stage_tag}.pt",
    )
    if not os.path.exists(stage_best):
        raise FileNotFoundError(
            f"Stage-best checkpoint not found for resume stage-best mode: {stage_best}"
        )
    _load_agent_state(agent, stage_best, label=f"stage_best({resume_stage_tag})")
    resumed_step = stage_start_steps_from_plan(curriculum_plan)[resume_stage_index]
    print(
        f"[Curriculum] resume stage-best mode: start {resume_stage_tag} from its "
        f"best checkpoint, reset optimizer state, set global_steps={resumed_step}"
    )
    return resumed_step


def load_init_agent_state(agent, init_agent_path):
    if not init_agent_path:
        return None
    if not os.path.exists(init_agent_path):
        raise FileNotFoundError(f"init_agent_path does not exist: {init_agent_path}")
    return _load_agent_state(agent, init_agent_path, label="init_agent")


def main(args):
    ckpt = resolve_ckpt_dir(args)
    accelerator = Accelerator(mixed_precision="bf16" if args.use_amp else "no")
    device = accelerator.device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    curriculum_plan = parse_curriculum_plan(args)
    total_planned_steps = sum(stage_steps for _, stage_steps in curriculum_plan)
    resume_from_stage = resolve_resume_from_stage(args, curriculum_plan)
    if args.resume_use_stage_best and args.resume_reset_stage_agent:
        raise ValueError(
            "resume_use_stage_best and resume_reset_stage_agent are mutually exclusive"
        )
    if args.resume_use_stage_best and resume_from_stage is None:
        raise ValueError(
            "resume_use_stage_best requires --resume-from-stage to be set"
        )
    if args.resume_load_agent_best_stage and args.resume_dir is None:
        raise ValueError(
            "resume_load_agent_best_stage requires --resume-dir so stage-best checkpoints "
            "can be resolved from the existing run directory"
        )

    probe_stage_mode = curriculum_plan[0][0]
    probe_env = gym.make(
        args.task_name,
        num_envs=1,
        **make_env_kwargs(probe_stage_mode, shader_dir="minimal", args=args),
    )
    agent_names = list(probe_env.unwrapped.agent.agents_dict.keys())
    probe_env.close()
    agent_obs_rules = build_agent_obs_rules(agent_names)

    collate_fn = make_collate_fn(device)
    infos = get_agent_info(
        args,
        make_env_kwargs(
            probe_stage_mode,
            shader_dir="default",
            args=args,
            include_sim_backend=False,
        ),
        agent_obs_rules,
        collate_fn,
    )
    agent = HeteroMAPPOAgent(
        agent_names=infos["agent_names"],
        state_dims=infos["state_dims"],
        global_state_dim=infos["global_state_dim"],
        action_dims=infos["action_dims"],
        camera_count=len(CAMERAS),
        normalize_state=args.normalize_state,
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    resumed_steps = 0
    if args.resume_dir is not None:
        if resume_from_stage is not None and args.resume_reset_stage_agent:
            resumed_steps = load_resume_from_previous_stage_best(
                agent, ckpt, curriculum_plan, resume_from_stage
            )
        elif resume_from_stage is not None and args.resume_use_stage_best:
            resumed_steps = load_resume_from_stage_best(
                agent, ckpt, curriculum_plan, resume_from_stage
            )
        else:
            resumed_steps = load_resume_state(agent, optimizer, ckpt)
    elif args.init_agent_path is not None:
        load_init_agent_state(agent, args.init_agent_path)
    if args.resume_load_agent_best_stage:
        load_multi_stage_agent_bests(
            agent,
            optimizer,
            ckpt,
            args.resume_load_agent_best_stage,
            agent_names,
        )

    agent, optimizer = accelerator.prepare(agent, optimizer)
    writer = SummaryWriter(ckpt["log_dir"]) if accelerator.is_main_process else None

    global_steps = int(resumed_steps)
    steps_per_rollout = args.num_envs * args.rollout_steps
    rollout_count = global_steps // max(steps_per_rollout, 1)
    best_score = -1.0
    metrics_log = load_json(ckpt["metrics"]) if os.path.exists(ckpt["metrics"]) else []
    final_stage_mode = curriculum_plan[-1][0]
    final_stage_scores = [
        float(row.get("score", -1.0))
        for row in metrics_log
        if normalize_stage_mode(row.get("stage", final_stage_mode)) == final_stage_mode
    ]
    if final_stage_scores:
        best_score = max(final_stage_scores)
    stage_best_scores = {}
    stage_gate_history = {}
    for row in metrics_log:
        stage_name = row.get("stage")
        if stage_name is None:
            continue
        stage_name = normalize_stage_mode(stage_name)
        stage_best_scores[stage_name] = max(
            float(row.get("score", -1.0)),
            stage_best_scores.get(stage_name, -1.0),
        )
        stage_gate_history[stage_name] = stage_gate_history.get(stage_name, False) or (
            float(row.get("score", -1.0)) >= float(args.curriculum_stage_success_threshold)
            and (
                float(args.curriculum_stage_success_episode_len_max) <= 0
                or (
                    np.isfinite(float(row.get("success_episode_len_mean", float("nan"))))
                    and float(row.get("success_episode_len_mean", float("nan")))
                    <= float(args.curriculum_stage_success_episode_len_max)
                )
            )
        )
    resume_stage_index = None
    if resume_from_stage is not None:
        resume_stage_index = next(
            idx for idx, (stage_mode, _) in enumerate(curriculum_plan)
            if stage_mode == resume_from_stage
        )
        forced_stage_modes = {
            stage_mode for stage_mode, _ in curriculum_plan[resume_stage_index:]
        }
        for forced_stage_mode in forced_stage_modes:
            stage_gate_history[forced_stage_mode] = False
        print(
            f"[Curriculum] force resume from {stage_tag(resume_from_stage)} "
            f"within resume_dir={args.resume_dir}; ignore historical unlocks for "
            f"{sorted(forced_stage_modes)}"
        )
    pbar = tqdm(total=total_planned_steps, initial=global_steps, ascii=True)
    last_entropy = float("nan")
    last_success_rate = float("nan")

    args.minibatch_size = (
        args.minibatch_size
        if args.minibatch_size > 0
        else (args.num_envs * args.rollout_steps) // args.num_minibatch
    )
    args.rollout_minibatch_size = 0
    args.value_huber_delta = 10.0
    args.full_kl_coef = 0.0
    args.log_full_kl = False
    args.zero_inactive_agents_during_rollout = True

    stage_start_steps = stage_start_steps_from_plan(curriculum_plan)
    stage_entropy_reset_done = set()

    for stage_index, (active_stage_mode, stage_total_steps) in enumerate(curriculum_plan):
        if resume_stage_index is not None and stage_index < resume_stage_index:
            continue
        stage_plan_start = stage_start_steps[stage_index]
        stage_plan_end = stage_plan_start + stage_total_steps
        if stage_total_steps <= 0:
            continue
        is_curriculum_gate_stage = (
            len(curriculum_plan) > 1 and stage_index < len(curriculum_plan) - 1
        )
        stage_success_threshold = (
            args.curriculum_stage_success_threshold if is_curriculum_gate_stage else 0.0
        )
        stage_success_episode_len_max = (
            args.curriculum_stage_success_episode_len_max if is_curriculum_gate_stage else 0.0
        )
        stage_min_evals = max(int(args.curriculum_stage_min_evals), 1)
        historical_stage_best = stage_best_scores.get(active_stage_mode, -1.0)
        active_stage_tag = stage_tag(active_stage_mode)
        if global_steps >= stage_plan_end:
            if is_curriculum_gate_stage and not stage_gate_history.get(active_stage_mode, False):
                print(
                    f"[Curriculum] resume continue at {active_stage_tag}: "
                    f"global_steps={global_steps} already passed planned_steps={stage_plan_end}, "
                    f"historical best_success={historical_stage_best:.3f} "
                    f"did not satisfy unlock gate; keep training this stage"
                )
            else:
                continue
        stage_gate_reached = (
            stage_success_threshold <= 0.0
            or stage_gate_history.get(active_stage_mode, False)
        )
        stage_eval_count = 0
        stage_rollouts_completed = max(0, global_steps - stage_plan_start) // max(steps_per_rollout, 1)
        explicit_critic_warmup = args.critic_warmup_rollouts is not None
        should_stage_warmup = stage_index > 0 or (
            stage_index == 0 and len(curriculum_plan) > 1 and active_stage_mode != "0"
        )
        if explicit_critic_warmup:
            stage_critic_warmup_rollouts = max(int(args.critic_warmup_rollouts), 0)
        else:
            stage_critic_warmup_rollouts = (
                args.stage_critic_warmup_rollouts if should_stage_warmup else 0
            )

        args.active_agent_names = get_stage_active_agent_names(active_stage_mode, agent_names)
        if args.reset_entropy and active_stage_mode not in stage_entropy_reset_done:
            reset_target_names = (
                agent_names if args.active_agent_names is None else args.active_agent_names
            )
            raw_agent = accelerator.unwrap_model(agent)
            reset_summary = {}
            for reset_name in reset_target_names:
                reset_summary[reset_name] = reset_agent_specific_entropy(raw_agent, reset_name)
            stage_entropy_reset_done.add(active_stage_mode)
            print(
                f"[Curriculum] reset entropy for {active_stage_tag}: "
                f"agents={reset_target_names}, reset_parts={reset_summary}"
            )
        args.critic_only_update = False
        if is_curriculum_gate_stage and stage_gate_history.get(active_stage_mode, False):
            print(
                f"[Curriculum] resume unlock {active_stage_tag} -> "
                f"{stage_tag(curriculum_plan[stage_index + 1][0])} "
                f"from historical unlock gate "
                f"(best_success={historical_stage_best:.3f}, "
                f"success_episode_len_max={stage_success_episode_len_max:.1f})"
            )
            continue
        env_kwargs = make_env_kwargs(active_stage_mode, shader_dir="minimal", args=args)
        eval_env_kwargs = make_env_kwargs(
            active_stage_mode,
            shader_dir="default",
            args=args,
            include_sim_backend=False,
        )
        eval_video_dir = (
            os.path.join(os.path.dirname(ckpt["video_dir"]), stage_video_tag(active_stage_mode))
            if len(curriculum_plan) > 1
            else f'{ckpt["video_dir"]}_train'
        )

        envs = gym.make(args.task_name, num_envs=args.num_envs, **env_kwargs)
        envs = FlattenRGBObservationWrapperForMARL(envs, agent_obs_rules=agent_obs_rules)
        envs = ManiSkillVectorEnv(
            envs,
            args.num_envs,
            ignore_terminations=args.ignore_partial_reset,
            record_metrics=True,
        )
        eval_envs = make_eval_envs(
            env_id=args.task_name,
            num_envs=args.num_eval_envs,
            sim_backend="gpu",
            env_kwargs=eval_env_kwargs,
            video_dir=eval_video_dir,
            wrappers=[
                lambda e: FlattenRGBObservationWrapperForMARL(
                    e, agent_obs_rules=agent_obs_rules
                )
            ],
        )

        next_obs, _ = envs.reset(seed=args.seed + stage_index)
        next_done = torch.zeros(args.num_envs, device=device)
        _, _ = eval_envs.reset(seed=args.seed + stage_index)
        stage_target_steps = (
            args.total_steps
            if is_curriculum_gate_stage and not stage_gate_reached
            else stage_plan_end
        )
        stage_budget_exhausted_logged = global_steps >= stage_plan_end
        if stage_critic_warmup_rollouts > 0 and stage_rollouts_completed < stage_critic_warmup_rollouts:
            print(
                f"[Curriculum] {active_stage_tag} critic warmup: "
                f"{stage_rollouts_completed}/{stage_critic_warmup_rollouts} rollouts completed"
            )

        while global_steps < stage_target_steps:
            args.critic_only_update = (
                stage_rollouts_completed < stage_critic_warmup_rollouts
            )
            if writer is not None:
                writer.add_scalar(
                    f"stage_training/{active_stage_tag}_critic_only_update",
                    float(args.critic_only_update),
                    global_steps,
                )
            rollout = collect_rollout(
                args=args,
                agent=agent,
                collate_fn=collate_fn,
                envs=envs,
                next_obs=next_obs,
                next_done=next_done,
                accelerator=accelerator,
                writer=writer,
                global_step=global_steps,
                clients=None,
            )
            (
                obs_buf,
                act_buf,
                logp_buf,
                rew_buf,
                done_buf,
                val_buf,
                final_val_buf,
                next_obs,
                next_done,
            ) = rollout

            adv_buf, ret_buf = compute_gae(
                rew_buf,
                done_buf,
                val_buf,
                final_val_buf,
                next_obs,
                next_done,
                agent,
                collate_fn,
                args,
                accelerator,
            )

            data = (
                obs_buf.reshape((-1,)),
                act_buf.reshape((-1,)),
                logp_buf.reshape((-1,)),
                adv_buf.reshape(-1),
                ret_buf.reshape(-1),
                val_buf.reshape(-1),
            )

            stats = mappo_update_on_policy(
                args,
                agent,
                optimizer,
                data,
                collate_fn,
                accelerator,
                f"MAPPO Pretrain [{active_stage_tag}]",
                writer,
                -1,
            )

            rollout_count += 1
            global_steps += steps_per_rollout
            stage_rollouts_completed += 1
            pbar.update(steps_per_rollout)
            last_entropy = float(stats["entropy"])
            if (
                stage_critic_warmup_rollouts > 0
                and stage_rollouts_completed == stage_critic_warmup_rollouts
            ):
                print(
                    f"[Curriculum] {active_stage_tag} critic warmup finished after "
                    f"{stage_rollouts_completed} rollouts; enable actor updates"
                )
            if (
                is_curriculum_gate_stage
                and not stage_gate_reached
                and not stage_budget_exhausted_logged
                and global_steps >= stage_plan_end
            ):
                stage_budget_exhausted_logged = True
                print(
                    f"[Curriculum] keep training {active_stage_tag} beyond planned_steps="
                    f"{stage_total_steps} because success threshold "
                    f"{stage_success_threshold:.3f} has not been reached yet"
                )
            pbar.set_postfix(
                stage=active_stage_tag,
                success=(
                    f"{last_success_rate:.3f}"
                    if not np.isnan(last_success_rate)
                    else "n/a"
                ),
                entropy=f"{last_entropy:.3f}",
            )

            if writer is not None:
                writer.add_scalar("loss/policy", stats["policy_loss"], global_steps)
                writer.add_scalar("loss/value", stats["value_loss"], global_steps)
                writer.add_scalar("loss/entropy", stats["entropy"], global_steps)
                writer.add_scalar("loss/approx_kl", stats["approx_kl"], global_steps)
                writer.add_scalar(f"stage_loss/{active_stage_tag}_policy", stats["policy_loss"], global_steps)
                writer.add_scalar(f"stage_loss/{active_stage_tag}_value", stats["value_loss"], global_steps)
                writer.add_scalar(f"stage_loss/{active_stage_tag}_entropy", stats["entropy"], global_steps)
                writer.add_scalar(f"stage_loss/{active_stage_tag}_approx_kl", stats["approx_kl"], global_steps)

            if rollout_count % args.eval_interval_rollouts == 0:
                agent.eval()
                eval_metrics = evaluate(
                    n=args.num_eval_episodes,
                    sample_fn=make_sample_fn(
                        agent,
                        collate_fn,
                        deterministic=True,
                        active_agent_names=args.active_agent_names,
                    ),
                    eval_envs=eval_envs,
                )
                score = extract_eval_success_score(eval_metrics)
                success_episode_len_mean = extract_eval_success_episode_len_mean(eval_metrics)
                stage_eval_count += 1
                last_success_rate = score
                metrics_log.append(
                    {
                        "step": global_steps,
                        "score": score,
                        "success_episode_len_mean": success_episode_len_mean,
                        "stage": active_stage_mode,
                        "stage_tag": active_stage_tag,
                    }
                )
                dump_json(ckpt["metrics"], metrics_log)
                pbar.set_postfix(
                    stage=active_stage_tag,
                    success=f"{last_success_rate:.3f}",
                    entropy=f"{last_entropy:.3f}",
                )
                if writer is not None:
                    for k, v in eval_metrics.items():
                        writer.add_scalar(f"eval/{k}", float(v.mean()), global_steps)
                        writer.add_scalar(
                            f"eval_{active_stage_tag}/{k}",
                            float(v.mean()),
                            global_steps,
                        )
                    writer.add_scalar(
                        f"eval_{active_stage_tag}/success_episode_len_mean",
                        success_episode_len_mean if np.isfinite(success_episode_len_mean) else np.nan,
                        global_steps,
                    )
                torch.save(agent.state_dict(), ckpt["latest_agent"])
                torch.save(
                    {"opt": optimizer.state_dict(), "step": global_steps},
                    ckpt["latest_opt"],
                )
                stage_best_path = os.path.join(
                    os.path.dirname(ckpt["best_agent"]),
                    f"best_agent_{active_stage_tag}.pt",
                )
                if score > stage_best_scores.get(active_stage_mode, -1.0):
                    stage_best_scores[active_stage_mode] = score
                    torch.save(agent.state_dict(), stage_best_path)
                if active_stage_mode == final_stage_mode and score > best_score:
                    best_score = score
                    torch.save(agent.state_dict(), ckpt["best_agent"])
                agent.train()
                if (
                    is_curriculum_gate_stage
                    and not stage_gate_reached
                    and stage_eval_count >= stage_min_evals
                    and stage_gate_satisfied(
                        score,
                        success_episode_len_mean,
                        stage_success_threshold,
                        stage_success_episode_len_max,
                    )
                ):
                    stage_gate_reached = True
                    stage_gate_history[active_stage_mode] = True
                    print(
                        f"[Curriculum] unlocked {active_stage_tag} -> "
                        f"{stage_tag(curriculum_plan[stage_index + 1][0])} "
                        f"with success_rate={score:.3f}, "
                        f"success_episode_len_mean={success_episode_len_mean:.2f} "
                        f"(threshold={stage_success_threshold:.3f}, "
                        f"len_max={stage_success_episode_len_max:.1f}, "
                        f"evals={stage_eval_count})"
                    )
                    break

        if is_curriculum_gate_stage and not stage_gate_reached:
            need_final_gate_eval = (
                stage_eval_count < stage_min_evals
                or np.isnan(last_success_rate)
            )
            if need_final_gate_eval:
                agent.eval()
                eval_metrics = evaluate(
                    n=args.num_eval_episodes,
                    sample_fn=make_sample_fn(
                        agent,
                        collate_fn,
                        deterministic=True,
                        active_agent_names=args.active_agent_names,
                    ),
                    eval_envs=eval_envs,
                )
                score = extract_eval_success_score(eval_metrics)
                success_episode_len_mean = extract_eval_success_episode_len_mean(eval_metrics)
                stage_eval_count += 1
                last_success_rate = score
                metrics_log.append(
                    {
                        "step": global_steps,
                        "score": score,
                        "success_episode_len_mean": success_episode_len_mean,
                        "stage": active_stage_mode,
                        "stage_tag": active_stage_tag,
                    }
                )
                dump_json(ckpt["metrics"], metrics_log)
                pbar.set_postfix(
                    stage=active_stage_tag,
                    success=f"{last_success_rate:.3f}",
                    entropy=f"{last_entropy:.3f}",
                )
                if writer is not None:
                    for k, v in eval_metrics.items():
                        writer.add_scalar(f"eval/{k}", float(v.mean()), global_steps)
                        writer.add_scalar(
                            f"eval_{active_stage_tag}/{k}",
                            float(v.mean()),
                            global_steps,
                        )
                    writer.add_scalar(
                        f"eval_{active_stage_tag}/success_episode_len_mean",
                        success_episode_len_mean if np.isfinite(success_episode_len_mean) else np.nan,
                        global_steps,
                    )
                torch.save(agent.state_dict(), ckpt["latest_agent"])
                torch.save(
                    {"opt": optimizer.state_dict(), "step": global_steps},
                    ckpt["latest_opt"],
                )
                stage_best_path = os.path.join(
                    os.path.dirname(ckpt["best_agent"]),
                    f"best_agent_{active_stage_tag}.pt",
                )
                if score > stage_best_scores.get(active_stage_mode, -1.0):
                    stage_best_scores[active_stage_mode] = score
                    torch.save(agent.state_dict(), stage_best_path)
                if stage_gate_satisfied(
                    score,
                    success_episode_len_mean,
                    stage_success_threshold,
                    stage_success_episode_len_max,
                ) and stage_eval_count >= stage_min_evals:
                    stage_gate_reached = True
                    stage_gate_history[active_stage_mode] = True
                    print(
                        f"[Curriculum] unlocked {active_stage_tag} -> "
                        f"{stage_tag(curriculum_plan[stage_index + 1][0])} "
                        f"with final success_rate={score:.3f}, "
                        f"success_episode_len_mean={success_episode_len_mean:.2f} "
                        f"(threshold={stage_success_threshold:.3f}, "
                        f"len_max={stage_success_episode_len_max:.1f}, "
                        f"evals={stage_eval_count})"
                    )
                agent.train()

        envs.close()
        eval_envs.close()
        args.critic_only_update = False
        if is_curriculum_gate_stage and not stage_gate_reached:
            print(
                f"[Curriculum] total_steps exhausted at {active_stage_tag}: "
                f"best_success={stage_best_scores.get(active_stage_mode, -1.0):.3f}, "
                f"last_success={last_success_rate:.3f}, "
                f"threshold={stage_success_threshold:.3f}, "
                f"planned_steps={stage_total_steps}"
            )
            break

    args.active_agent_names = None
    torch.save(agent.state_dict(), ckpt["latest_agent"])
    torch.save({"opt": optimizer.state_dict(), "step": global_steps}, ckpt["latest_opt"])
    if writer is not None:
        writer.close()
    pbar.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="PassObjectFiveRobots-v1")
    parser.add_argument("--robot-name", type=str, default="panda_so100_widowx_xarm6_inspire")
    parser.add_argument("--seed", type=int, default=1788)
    parser.add_argument("--total-steps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-eval-envs", type=int, default=8)
    parser.add_argument("--num-eval-episodes", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatch", type=int, default=32)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.956)
    parser.add_argument("--gae-lambda", type=float, default=0.966)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--clip-vloss", action="store_true")
    parser.add_argument("--finite-horizon-gae", action="store_true")
    parser.add_argument("--normalize-state", action="store_true")
    parser.add_argument("--not-normalize-adv", action="store_true")
    parser.add_argument("--ignore-partial-reset", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--save-dir", type=str, default="ckpt")
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--resume-from-stage", type=str, default=None)
    parser.add_argument("--resume-use-stage-best", action="store_true")
    parser.add_argument("--resume-reset-stage-agent", action="store_true")
    parser.add_argument("--resume-load-agent-best-stage", type=str, default=None)
    parser.add_argument("--reset-entropy", action="store_true")
    parser.add_argument("--init-agent-path", type=str, default=None)
    parser.add_argument("--eval-interval-rollouts", type=int, default=3)
    parser.add_argument("--stage-mode", type=str, default="full")
    parser.add_argument("--stage-max-episode-steps", type=int, default=60)
    parser.add_argument("--post-stage0-max-episode-steps", type=int, default=100)
    parser.add_argument("--full-max-episode-steps", type=int, default=300)
    parser.add_argument("--curriculum-stages", type=str, default="")
    parser.add_argument("--curriculum-stage-steps", type=str, default="")
    parser.add_argument("--curriculum-stage-success-threshold", type=float, default=0.6)
    parser.add_argument("--curriculum-stage-success-episode-len-max", type=float, default=60.0)
    parser.add_argument("--curriculum-stage-min-evals", type=int, default=1)
    parser.add_argument("--critic-warmup-rollouts", type=int, default=None)
    parser.add_argument("--stage-critic-warmup-rollouts", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
