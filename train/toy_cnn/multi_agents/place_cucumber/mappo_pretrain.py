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
import envs.place_cucumber  # noqa: F401


MODEL_NAME = "toy_cnn_place_cucumber_mappo_pretrain"
CAMERAS = ("base_camera",)


def set_critic_warmup_trainable(agent, enabled: bool):
    module = agent.module if hasattr(agent, "module") else agent
    critic_prefixes = ("critic_state_encoder", "critic")
    for name, param in module.named_parameters():
        param.requires_grad_(not enabled or name.startswith(critic_prefixes))


class FlattenRGBObservationWrapperForMARL(gym.ObservationWrapper):
    def __init__(self, env, agent_obs_rules, rgb=True, state=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_state = state
        self.agent_obs_rules = agent_obs_rules

        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "rgb" not in first_cam:
            self.include_rgb = False
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

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
            agent_states[agent_name] = common.flatten_state_dict(
                selected, use_torch=True, device=self.base_env.device
            )

        global_state = common.flatten_state_dict(
            observation, use_torch=True, device=self.base_env.device
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


def make_sample_fn(agent, collate_fn, deterministic=True):
    def sample_fn(obs):
        return agent.get_action(collate_fn(obs), deterministic=deterministic)

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


def main(args):
    ckpt = resolve_ckpt_dir(args)
    accelerator = Accelerator(mixed_precision="bf16" if args.use_amp else "no")
    device = accelerator.device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kwargs = {
        "obs_mode": "rgb+state_dict",
        "control_mode": "pd_joint_delta_pos",
        "render_mode": "rgb_array",
        "reward_mode": "normalized_dense",
        "shader_dir": "minimal",
        "sim_backend": "physx_cuda",
    }
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs.pop("sim_backend")
    eval_env_kwargs["shader_dir"] = "default"

    probe_env = gym.make(args.task_name, num_envs=1, **env_kwargs)
    agent_names = list(probe_env.unwrapped.agent.agents_dict.keys())
    probe_env.close()

    center_name, left_name, right_name = agent_names
    agent_obs_rules = {
        center_name: [
            "qpos",
            "qvel",
            "center_tcp",
            "center_tcp_q",
            "lid_pos",
            "lid_handle_pos",
            "lid_qpos",
            "lid_qvel",
            "center_tcp_to_lid_handle",
        ],
        left_name: [
            "qpos",
            "qvel",
            "left_tcp",
            "left_tcp_q",
            "left_grip_line_axis",
            "cucumber1_pos",
            "cucumber1_q",
            "cucumber1_long_axis",
            "pot_pos",
            "left_tcp_to_cucumber1",
            "cucumber1_to_pot",
        ],
        right_name: [
            "qpos",
            "qvel",
            "right_tcp",
            "right_tcp_q",
            "right_grip_line_axis",
            "cucumber2_pos",
            "cucumber2_q",
            "cucumber2_long_axis",
            "pot_pos",
            "right_tcp_to_cucumber2",
            "cucumber2_to_pot",
        ],
    }

    collate_fn = make_collate_fn(device)
    infos = get_agent_info(args, eval_env_kwargs, agent_obs_rules, collate_fn)
    agent = HeteroMAPPOAgent(
        agent_names=infos["agent_names"],
        state_dims=infos["state_dims"],
        global_state_dim=infos["global_state_dim"],
        action_dims=infos["action_dims"],
        camera_count=len(CAMERAS),
        normalize_state=args.normalize_state,
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    resume_agent_path = ckpt["best_agent"] if args.resume_best_agent else ckpt["latest_agent"]
    if os.path.exists(resume_agent_path):
        agent.load_state_dict(torch.load(resume_agent_path, map_location="cpu"))
        print(f"[MAPPO Pretrain] Loaded agent checkpoint: {resume_agent_path}")
    if os.path.exists(ckpt["latest_opt"]) and not args.skip_load_opt:
        optimizer.load_state_dict(torch.load(ckpt["latest_opt"], map_location="cpu")["opt"])

    agent, optimizer = accelerator.prepare(agent, optimizer)
    writer = SummaryWriter(ckpt["log_dir"]) if accelerator.is_main_process else None

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
        video_dir=f'{ckpt["video_dir"]}_train',
        wrappers=[lambda e: FlattenRGBObservationWrapperForMARL(e, agent_obs_rules=agent_obs_rules)],
    )

    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    _, _ = eval_envs.reset(seed=args.seed)

    global_steps = 0
    rollout_count = 0
    best_score = -1.0
    metrics_log = load_json(ckpt["metrics"]) if os.path.exists(ckpt["metrics"]) else []
    pbar = tqdm(total=args.total_steps, ascii=True)
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
    print(f"[MAPPO Pretrain] critic_warmup_rollouts={args.critic_warmup_rollouts}")

    while global_steps < args.total_steps:
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

        critic_only_update = rollout_count < args.critic_warmup_rollouts
        args.critic_only_update = critic_only_update
        set_critic_warmup_trainable(agent, critic_only_update)
        stats = mappo_update_on_policy(
            args,
            agent,
            optimizer,
            data,
            collate_fn,
            accelerator,
            "MAPPO Pretrain",
            writer,
            -1,
        )
        if critic_only_update:
            set_critic_warmup_trainable(agent, False)

        rollout_count += 1
        global_steps += args.num_envs * args.rollout_steps
        pbar.update(args.num_envs * args.rollout_steps)
        last_entropy = float(stats["entropy"])
        pbar.set_postfix(
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

        if rollout_count % args.eval_interval_rollouts == 0:
            agent.eval()
            eval_metrics = evaluate(
                n=args.num_eval_episodes,
                sample_fn=make_sample_fn(agent, collate_fn, deterministic=True),
                eval_envs=eval_envs,
            )
            score = float(
                eval_metrics.get("success_rate", eval_metrics[list(eval_metrics.keys())[0]]).mean()
            )
            last_success_rate = score
            metrics_log.append({"step": global_steps, "score": score})
            dump_json(ckpt["metrics"], metrics_log)
            pbar.set_postfix(
                success=f"{last_success_rate:.3f}",
                entropy=f"{last_entropy:.3f}",
            )
            if writer is not None:
                for k, v in eval_metrics.items():
                    writer.add_scalar(f"eval/{k}", float(v.mean()), global_steps)
            torch.save(agent.state_dict(), ckpt["latest_agent"])
            torch.save({"opt": optimizer.state_dict(), "step": global_steps}, ckpt["latest_opt"])
            if score >= best_score:
                best_score = score
                torch.save(agent.state_dict(), ckpt["best_agent"])
            agent.train()

    torch.save(agent.state_dict(), ckpt["latest_agent"])
    torch.save({"opt": optimizer.state_dict(), "step": global_steps}, ckpt["latest_opt"])
    if writer is not None:
        writer.close()
    envs.close()
    eval_envs.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="PlaceCucumber-v1")
    parser.add_argument("--robot-name", type=str, default="panda_widowx_widowx")
    parser.add_argument("--seed", type=int, default=1788)
    parser.add_argument("--total-steps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
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
    parser.add_argument("--resume-best-agent", action="store_true")
    parser.add_argument("--skip-load-opt", action="store_true")
    parser.add_argument("--eval-interval-rollouts", type=int, default=3)
    parser.add_argument("--critic-warmup-rollouts", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
