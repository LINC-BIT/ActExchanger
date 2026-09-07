import os
import sys
import time
from datetime import datetime

sys.path.append(os.getcwd())

from train.toy_cnn.multi_agents.two_robot_pick.gpu_auto_select import configure_cuda_visible_devices

configure_cuda_visible_devices()

import argparse
import random

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.io_utils import dump_json, load_json
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch import nn
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import envs.two_robot_stack_cube_v1  # noqa: F401
from train.marl.mappo.base import collect_rollout, mappo_update_on_policy
from train.reinforcement_learning.evaluate import evaluate
from train.reinforcement_learning.make_env import make_eval_envs
from train.reinforcement_learning.utils import RunningMeanStd, compute_gae, get_stage, get_step_infos
from train.toy_cnn.model import PlainConv, make_mlp, make_mlp_with_orth_init

mp.set_start_method("spawn", force=True)


MODEL_NAME = "toy_cnn"
ROBOT_NAME = "panda_ur10e_panda_gripper"


class FlattenRGBObservationWrapperForHeteroMARL(gym.ObservationWrapper):
    def __init__(self, env, agent_obs_rules, rgb=True, state=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_state = state

        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "rgb" not in first_cam:
            self.include_rgb = False
        self.agent_obs_rules = agent_obs_rules
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: dict):
        observation = dict(observation)
        sensor_data = observation.pop("sensor_data")
        observation.pop("sensor_param", None)

        rgb_images = {}
        for cam_name, cam_data in sensor_data.items():
            if self.include_rgb:
                rgb_images[cam_name] = cam_data["rgb"]

        agent_states = {}
        for agent_name in observation["agent"].keys():
            observation_states = {}
            for key in self.agent_obs_rules[agent_name]:
                if key in observation["agent"][agent_name]:
                    observation_states[key] = observation["agent"][agent_name][key].to(self.base_env.device)
                elif key in observation["extra"]:
                    observation_states[key] = observation["extra"][key].to(self.base_env.device)
                else:
                    raise ValueError(f"Key {key} not found in agent obs or extra obs for {agent_name}")
            agent_states[agent_name] = common.flatten_state_dict(
                observation_states, use_torch=True, device=self.base_env.device
            )

        global_state = common.flatten_state_dict(
            observation, use_torch=True, device=self.base_env.device
        )

        ret = {}
        if self.include_rgb:
            assert len(rgb_images) == 1
            ret["rgb"] = rgb_images["base_camera"]
        if self.include_state:
            ret["agent_states"] = agent_states
            for agent_name, state in agent_states.items():
                ret[f"agent_states_{agent_name}"] = state
            ret["global_state"] = global_state
        return ret


class HeteroMAPPOAgent(nn.Module):
    def __init__(self, agent_names, state_dims, action_dims, global_state_dim, camera_count=1, normalize_state=True):
        super().__init__()
        self.agent_names = list(agent_names)
        self.state_dims = dict(state_dims)
        self.action_dims = dict(action_dims)

        self.rgb_encoder = PlainConv(
            in_channels=3 * camera_count,
            out_dim=256,
            max_pooling=False,
            inactivated_output=False,
        )

        if normalize_state:
            self.actor_state_rms = nn.ModuleDict(
                {
                    name: RunningMeanStd(shape=(self.state_dims[name],))
                    for name in self.agent_names
                }
            )
            self.critic_state_rms = RunningMeanStd(shape=(global_state_dim,))
        else:
            self.actor_state_rms = None
            self.critic_state_rms = None

        self.actor_state_encoders = nn.ModuleDict(
            {
                name: make_mlp(self.state_dims[name], [256, 256], last_act=False)
                for name in self.agent_names
            }
        )
        self.critic_state_encoder = make_mlp(global_state_dim, [512, 512], last_act=False)

        self.actor_heads = nn.ModuleDict()
        self.actor_logstd = nn.ParameterDict()
        self.actor_feature_placeholders = nn.ModuleDict()
        for name in self.agent_names:
            self.actor_heads[name] = make_mlp_with_orth_init(
                512, [512, self.action_dims[name]], last_act=False, is_actor=True
            )
            self.actor_logstd[name] = nn.Parameter(torch.ones(1, self.action_dims[name]) * -0.5)
            self.actor_feature_placeholders[name] = nn.Identity()

        self.critic = make_mlp_with_orth_init(768, [512, 1], last_act=False)

    def get_actor_feature(self, rgb_feat, agent_state_feat):
        return torch.cat([rgb_feat, agent_state_feat], dim=1)

    def get_critic_feature(self, rgb_feat, global_state):
        global_feat = self.critic_state_encoder(global_state)
        return torch.cat([rgb_feat, global_feat], dim=1)

    def get_action_and_value(self, batch, actions_input=None, return_token_logits=False):
        rgb = batch["rgb"]
        global_state = batch["global_state"]
        rgb_feat = self.rgb_encoder(rgb)

        if self.critic_state_rms is not None:
            global_state = self.critic_state_rms(global_state)
        critic_feat = self.get_critic_feature(rgb_feat, global_state)
        value = self.critic(critic_feat).squeeze(-1)

        actions_out = {}
        log_probs = {}
        entropies = {}
        for name in self.agent_names:
            agent_state = batch[f"agent_states_{name}"]
            if self.actor_state_rms is not None:
                agent_state = self.actor_state_rms[name](agent_state)
            state_feat = self.actor_state_encoders[name](agent_state)
            actor_feat = self.get_actor_feature(rgb_feat, state_feat)
            actor_feat = self.actor_feature_placeholders[name](actor_feat)
            mean = self.actor_heads[name](actor_feat)
            logstd = self.actor_logstd[name].expand_as(mean)
            std = torch.exp(logstd)
            dist = Normal(mean, std)
            if actions_input is None:
                action = dist.sample()
            else:
                action = actions_input[name]
            if not torch.is_tensor(action):
                action = torch.tensor(action, dtype=mean.dtype, device=mean.device)
            actions_out[name] = action.detach().cpu().numpy()
            log_probs[name] = dist.log_prob(action).sum(-1)
            entropies[name] = dist.entropy().sum(-1)
        if return_token_logits:
            return actions_out, log_probs, entropies, value, None
        return actions_out, log_probs, entropies, value

    @torch.no_grad()
    def get_action(self, batch, deterministic=False):
        rgb = batch["rgb"]
        rgb_feat = self.rgb_encoder(rgb)
        actions = {}
        for name in self.agent_names:
            state = batch[f"agent_states_{name}"]
            if self.actor_state_rms is not None:
                state = self.actor_state_rms[name](state)
            state_feat = self.actor_state_encoders[name](state)
            feat = self.get_actor_feature(rgb_feat, state_feat)
            feat = self.actor_feature_placeholders[name](feat)
            mean = self.actor_heads[name](feat)
            if deterministic:
                actions[name] = mean
            else:
                std = torch.exp(self.actor_logstd[name].expand_as(mean))
                dist = Normal(mean, std)
                actions[name] = dist.sample()
        return actions

    def get_value(self, batch):
        rgb = batch["rgb"]
        global_state = batch["global_state"]
        if self.critic_state_rms is not None:
            global_state = self.critic_state_rms(global_state)
        rgb_feat = self.rgb_encoder(rgb)
        critic_feat = self.get_critic_feature(rgb_feat, global_state)
        return self.critic(critic_feat).squeeze(-1)

    @torch.no_grad()
    def update_state_stats(self, obs):
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].update(obs[f"agent_states_{name}"])
        if self.critic_state_rms is not None:
            self.critic_state_rms.update(obs["global_state"])

    def freeze_state_stats(self):
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].freeze()
        if self.critic_state_rms is not None:
            self.critic_state_rms.freeze()

    def unfreeze_state_stats(self):
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].unfreeze()
        if self.critic_state_rms is not None:
            self.critic_state_rms.unfreeze()

    def reset_logstd(self, new_logstd=-0.5):
        for name in self.agent_names:
            self.actor_logstd[name].data.fill_(new_logstd)


def resolve_ckpt_dir(args):
    task_dir = os.path.join(args.save_dir, f"{args.task_name}/ppo/{args.robot_name}/{MODEL_NAME}")
    root_dir = os.path.join(task_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(task_dir, exist_ok=True)
    if args.resume_dir is not None:
        root_dir = args.resume_dir
    log_dir = os.path.join(root_dir, "tb")
    video_dir = os.path.join(root_dir, "videos")
    return {
        "task_dir": task_dir,
        "log_dir": log_dir,
        "root_dir": root_dir,
        "video_dir": video_dir,
        "latest_agent": os.path.join(root_dir, "latest_agent.pt"),
        "latest_opt": os.path.join(root_dir, "latest_opt.pt"),
        "best_agent": os.path.join(root_dir, "best_agent.pt"),
        "metrics": os.path.join(root_dir, "metrics.json"),
    }


def make_collate_fn(device):
    def _resize(img, size=128):
        return F.interpolate(img, size=size, mode="bilinear")

    def collate_fn(obs):
        if isinstance(obs["rgb"], np.ndarray):
            rgb = torch.from_numpy(obs["rgb"] / 255.0).permute(0, 3, 1, 2).float()
            global_state = torch.from_numpy(obs["global_state"])
            agent_states = {name: torch.from_numpy(state) for name, state in obs["agent_states"].items()}
        else:
            rgb = obs["rgb"].permute(0, 3, 1, 2).float() / 255.0
            global_state = obs["global_state"]
            agent_states = obs["agent_states"]

        batch = {"rgb": _resize(rgb).to(device), "global_state": global_state.to(device), "agent_states": {}}
        for name, state in agent_states.items():
            state = state.to(device)
            batch["agent_states"][name] = state
            batch[f"agent_states_{name}"] = state
        return batch

    return collate_fn


def make_sample_fn(agent, accelerator, deterministic=True):
    collate_fn = make_collate_fn(accelerator.device)

    def sample_fn(obs):
        return agent.get_action(collate_fn(obs), deterministic=deterministic)

    return sample_fn


def get_agent_info(args, env_kwargs, agent_obs_rules, device):
    test_env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend="gpu",
        env_kwargs=env_kwargs,
        wrappers=[lambda env: FlattenRGBObservationWrapperForHeteroMARL(env, agent_obs_rules=agent_obs_rules)],
    )
    obs, _ = test_env.reset()
    batch = make_collate_fn(device)(obs)
    agent_names = list(batch["agent_states"].keys())
    state_dims = {agent_name: int(batch["agent_states"][agent_name].shape[-1]) for agent_name in agent_names}
    action_dims = {
        agent_name: int(test_env.unwrapped.agent.agents_dict[agent_name].single_action_space.shape[0])
        for agent_name in agent_names
    }
    global_state_dim = int(batch["global_state"].shape[-1])
    test_env.close()
    return {
        "agent_names": agent_names,
        "state_dims": state_dims,
        "action_dims": action_dims,
        "global_state_dim": global_state_dim,
    }


def main(args):
    ckpt = resolve_ckpt_dir(args)
    step_infos = get_step_infos(args)

    env_kwargs = {
        "obs_mode": "rgb+state_dict",
        "control_mode": "pd_joint_delta_pos",
        "render_mode": "rgb_array",
        "reward_mode": "normalized_dense",
        "shader_dir": "minimal",
        "sim_backend": "physx_cuda",
        "max_episode_steps": args.max_episode_steps,
    }
    env_kwargs_for_eval = dict(env_kwargs)
    env_kwargs_for_eval.pop("sim_backend")
    env_kwargs["render_mode"] = "none"
    env_kwargs_for_eval["shader_dir"] = "default"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = not args.ignore_torch_deterministic

    accelerator = Accelerator(mixed_precision="bf16" if args.use_amp else "no")
    device = accelerator.device

    agent_obs_rules = {
        "panda_wristcam-0": [
            "cubeA_pose",
            "cubeB_pose",
            "left_arm_tcp_to_cubeA_pos",
            "left_arm_tcp",
            "qpos",
            "qvel",
            "stage",
        ],
        "ur10e_panda_gripper-1": [
            "cubeA_pose",
            "cubeB_pose",
            "right_arm_tcp_to_cubeB_pos",
            "right_arm_tcp",
            "qpos",
            "qvel",
            "stage",
        ],
    }

    infos = get_agent_info(args, env_kwargs_for_eval, agent_obs_rules, device)
    agent = HeteroMAPPOAgent(
        agent_names=infos["agent_names"],
        state_dims=infos["state_dims"],
        action_dims=infos["action_dims"],
        global_state_dim=infos["global_state_dim"],
        normalize_state=args.normalize_state,
    )

    if os.path.exists(ckpt["latest_agent"]):
        print(f"[Train] Resume agent from {ckpt['latest_agent']}")
        agent.load_state_dict(torch.load(ckpt["latest_agent"], map_location="cpu"))

    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    global_steps = sta_steps = 0
    start_time = time.time()
    if os.path.exists(ckpt["latest_opt"]):
        print(f"[Train] Resume optimizer from {ckpt['latest_opt']}")
        resume_opt = torch.load(ckpt["latest_opt"], map_location="cpu")
        optimizer.load_state_dict(resume_opt["opt"])
        global_steps = sta_steps = int(resume_opt["step"])

    agent, optimizer = accelerator.prepare(agent, optimizer)
    if args.reset_logstd:
        agent.reset_logstd(-0.5)

    if args.minibatch_size == 0:
        args.minibatch_size = step_infos["rollot_steps"] // args.num_minibatch // args.grad_accum_steps

    eval_envs = make_eval_envs(
        env_id=args.task_name,
        num_envs=args.num_eval_envs,
        sim_backend="gpu",
        env_kwargs=env_kwargs_for_eval,
        video_dir=f"{ckpt['video_dir']}_eval",
        wrappers=[lambda env: FlattenRGBObservationWrapperForHeteroMARL(env, agent_obs_rules=agent_obs_rules)],
    )
    envs = gym.make(args.task_name, num_envs=args.num_envs, **env_kwargs)
    envs = FlattenRGBObservationWrapperForHeteroMARL(envs, agent_obs_rules=agent_obs_rules)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=args.ignore_partial_reset, record_metrics=True)
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    best_score = -1.0
    metrics_log = load_json(ckpt["metrics"]) if os.path.exists(ckpt["metrics"]) else []
    resume_skip = args.resume_dir is not None

    if args.evaluate_mode:
        if args.eval_agent_dir is None:
            raise ValueError("Please provide --eval-agent-dir for evaluation mode")
        agent.load_state_dict(torch.load(os.path.join(args.eval_agent_dir, "best_agent.pt"), map_location="cpu"))
        agent = accelerator.prepare(agent)
        agent.eval()
        eval_metrics = evaluate(
            n=100,
            sample_fn=make_sample_fn(agent, accelerator, deterministic=True),
            eval_envs=eval_envs,
        )
        payload = {k: float(v.mean()) for k, v in eval_metrics.items()}
        for k, v in eval_metrics.items():
            print(f"eval_{k}_mean={v.mean()}")
        dump_json(os.path.join(args.eval_agent_dir, "eval_metrics.json"), payload)
        return

    writer = SummaryWriter(ckpt["log_dir"], purge_step=global_steps)
    print(f"[TensorBoard] Logging to {ckpt['log_dir']}")
    pbar = tqdm(total=step_infos["total_steps"], initial=global_steps, ascii=True)

    while global_steps < step_infos["total_steps"]:
        agent.eval()
        if not resume_skip and global_steps % step_infos["save_interval_steps"] == 0:
            torch.save(agent.state_dict(), ckpt["latest_agent"])
            torch.save({"opt": optimizer.state_dict(), "step": global_steps}, ckpt["latest_opt"])
            eval_metrics = evaluate(
                n=1,
                sample_fn=make_sample_fn(agent, accelerator, deterministic=True),
                eval_envs=eval_envs,
            )
            for key, value in eval_metrics.items():
                mean = value.mean()
                writer.add_scalar(f"eval/{key}", mean, global_steps)
                print(f"eval_{key}_mean={mean}")
            score = float(eval_metrics.get("success_rate", eval_metrics[list(eval_metrics.keys())[0]]).mean())
            pbar.set_postfix(eval_score=score)
            metrics_log.append(dict(step=global_steps, score=score))
            dump_json(ckpt["metrics"], metrics_log)
            if score >= best_score:
                best_score = score
                torch.save(agent.state_dict(), ckpt["best_agent"])
                print(f"[Eval] New best model saved (score={score:.3f})")

        resume_skip = False

        if args.dynamic_clip:
            frac = 1.0 - (global_steps / step_infos["total_steps"])
            args.clip_eps = 0.1 + (args.clip_eps - 0.1) * frac
        if args.dynamic_ent_coef:
            frac = 1.0 - (global_steps / step_infos["total_steps"])
            args.ent_coef = args.ent_coef / 10 + (args.ent_coef * 0.9) * frac

        collate_fn = make_collate_fn(device)
        rollout = collect_rollout(
            args=args,
            agent=agent,
            collate_fn=collate_fn,
            accelerator=accelerator,
            envs=envs,
            next_obs=next_obs,
            next_done=next_done,
            writer=writer,
            global_step=global_steps,
        )

        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, final_val_buf, next_obs, next_done = rollout
        adv_buf, ret_buf = compute_gae(
            rew_buf, done_buf, val_buf, final_val_buf, next_obs, next_done, agent, collate_fn, args, accelerator
        )

        data = (
            obs_buf.reshape((-1,)),
            act_buf.reshape((-1,)),
            logp_buf.reshape((-1,)),
            adv_buf.reshape(-1),
            ret_buf.reshape(-1),
            val_buf.reshape(-1),
        )

        agent.train()
        stats = mappo_update_on_policy(
            args, agent, optimizer, data, collate_fn, accelerator, get_stage(global_steps, step_infos), writer, -1
        )

        global_steps += step_infos["rollot_steps"]
        pbar.update(step_infos["rollot_steps"])

        y_pred = val_buf.flatten(0, 1).cpu().numpy()
        y_true = ret_buf.flatten(0, 1).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = (global_steps - sta_steps) / max(time.time() - start_time, 1e-6)
        writer.add_scalar("charts/SPS", sps, global_steps)
        writer.add_scalar("loss/policy", stats["policy_loss"], global_steps)
        writer.add_scalar("loss/value", stats["value_loss"], global_steps)
        writer.add_scalar("loss/entropy", stats["entropy"], global_steps)
        writer.add_scalar("loss/approx_kl", stats["approx_kl"], global_steps)
        writer.add_scalar("loss/old_approx_kl", stats["old_approx_kl"], global_steps)
        writer.add_scalar("loss/clip_frac", stats["clip_frac"], global_steps)
        writer.add_scalar("loss/explained_var", explained_var, global_steps)

    writer.close()
    torch.save(agent.state_dict(), ckpt["latest_agent"])
    torch.save({"opt": optimizer.state_dict(), "step": global_steps}, ckpt["latest_opt"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="TwoRobotStackCubeUR10e-v1")
    parser.add_argument("--seed", type=int, default=1788)
    parser.add_argument("--total-steps", type=int, default=50_000_000)
    parser.add_argument("--critic-warmup-rollouts", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-eval-envs", type=int, default=8)
    parser.add_argument("--ignore-partial-reset", action="store_true")
    parser.add_argument("--ignore-torch-deterministic", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num_minibatch", type=int, default=32)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--reward-scale", type=int, default=1)
    parser.add_argument("--rollout-minibatch-size", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--dynamic-clip", action="store_true")
    parser.add_argument("--dynamic-ent-coef", action="store_true")
    parser.add_argument("--normalize-state", action="store_true")
    parser.add_argument("--not-normalize-adv", action="store_true")
    parser.add_argument("--reset-logstd", action="store_true")
    parser.add_argument("--finite-horizon-gae", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.956)
    parser.add_argument("--gae-lambda", type=float, default=0.966)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--clip-vloss", action="store_true")
    parser.add_argument("--value-huber-delta", type=float, default=10.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.2)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--save-dir", type=str, default="ckpt")
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--save-interval-per-rollout", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--evaluate-mode", action="store_true")
    parser.add_argument("--eval-agent-dir", type=str, default=None)
    parser.add_argument("--robot-name", type=str, default=ROBOT_NAME)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
