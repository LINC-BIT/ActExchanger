import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

sys.path.append(os.getcwd())

from train.toy_cnn.multi_agents.two_robot_pick.gpu_auto_select import configure_cuda_visible_devices

configure_cuda_visible_devices()

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import gymnasium as gym
from mani_skill.utils.io_utils import dump_json, load_json
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import envs.place_cucumber  # noqa: F401
from train.reinforcement_learning.evaluate import evaluate
from train.reinforcement_learning.make_env import make_eval_envs
from train.toy_cnn.multi_agents.place_cucumber.mappo_pretrain import (
    FlattenRGBObservationWrapperForMARL as ToyCNNObservationWrapper,
    get_agent_info as get_toy_cnn_agent_info,
    make_collate_fn as make_toy_cnn_collate_fn,
)
from train.toy_cnn.multi_agents.place_cucumber.model import HeteroMAPPOAgent as ToyCNNMAPPOAgent
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.mixed_agent import (
    MixedTinyVLAAdapterSmolVLAEfficientVLAAgent,
)
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.model import (
    build_batch_from_obs,
)

mp.set_start_method("spawn", force=True)


MODEL_NAME = "vla_adapter_smolvla_efficientvla_sft"
ALGO_NAME = "sft"


def get_model_name(args) -> str:
    if args.model_backbone == "mixed_tiny_vla_smolvla_efficientvla":
        return MODEL_NAME
    raise ValueError(f"Unsupported model_backbone={args.model_backbone}")


def resolve_ckpt_dir(args) -> Dict[str, str]:
    task_dir = os.path.join(args.save_dir, f"{args.task_name}/{ALGO_NAME}/{args.robot_name}/{get_model_name(args)}")
    root_dir = os.path.join(task_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(task_dir, exist_ok=True)
    dataset_dir = os.path.join("datasets", args.task_name, "rl")
    os.makedirs(dataset_dir, exist_ok=True)

    resume_dir = args.resume_dir if args.resume_dir is not None else root_dir
    return {
        "task_dir": task_dir,
        "root_dir": root_dir,
        "log_dir": os.path.join(resume_dir, "tb"),
        "video_dir": os.path.join(resume_dir, "videos"),
        "latest_agent": os.path.join(resume_dir, "latest_agent.pt"),
        "latest_opt": os.path.join(resume_dir, "latest_opt.pt"),
        "best_agent": os.path.join(resume_dir, "best_agent.pt"),
        "metrics": os.path.join(resume_dir, "metrics.json"),
        "trajectory_h5": os.path.join(
            dataset_dir,
            f"trajectory.{args.obs_mode}.{args.control_mode}.physx_cuda.h5",
        ),
        "dataset_cache": os.path.join(resume_dir, "expert_sft_dataset.pt"),
    }


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def create_env_kwargs(args, render_mode: str, sim_backend: str | None) -> Dict[str, Any]:
    kwargs = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": render_mode,
        "max_episode_steps": args.max_episode_steps,
    }
    if sim_backend is not None:
        kwargs["sim_backend"] = sim_backend
    return kwargs


def make_sample_fn(agent_names: Sequence[str], agent, deterministic: bool = True):
    def sample_fn(obs):
        batch = build_batch_from_obs(obs, list(agent_names))
        return agent.get_action(batch, deterministic=deterministic)

    return sample_fn


def build_toy_cnn_agent_obs_rules(agent_names: Sequence[str]) -> Dict[str, list[str]]:
    center_name, left_name, right_name = list(agent_names)
    return {
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


def get_agent_info(args) -> Dict[str, Any]:
    env_kwargs = create_env_kwargs(args, render_mode="rgb_array", sim_backend=None)
    env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend="gpu",
        env_kwargs=env_kwargs,
    )
    obs, _ = env.reset()
    agent_names = list(obs["agent"].keys())
    batch = build_batch_from_obs(obs, agent_names)
    info = {
        "agent_names": agent_names,
        "state_dim": batch[f"agent_states_{agent_names[0]}"].shape[-1],
        "state_dims": {name: batch[f"agent_states_{name}"].shape[-1] for name in agent_names},
        "global_state_dim": batch["global_state"].shape[-1],
        "action_dim": env.single_action_space[agent_names[0]].shape[0],
    }
    env.close()
    return info


def _checkpoint_agent_names(state_dict: Dict[str, torch.Tensor]) -> list[str]:
    names = set()
    prefixes = (
        "actor_state_rms.",
        "actor_state_encoders.",
        "actor_heads.",
        "actor_logstd.",
    )
    for key in state_dict.keys():
        for prefix in prefixes:
            if key.startswith(prefix):
                names.add(key[len(prefix) :].split(".", 1)[0])
    return sorted(names)


def resolve_toy_cnn_expert_candidates(args) -> list[str]:
    if args.expert_agent_path:
        if not os.path.exists(args.expert_agent_path):
            raise FileNotFoundError(f"Missing expert agent checkpoint: {args.expert_agent_path}")
        return [args.expert_agent_path]

    expert_root = os.path.join(
        args.save_dir,
        f"{args.task_name}/ppo/{args.robot_name}/toy_cnn_place_cucumber_mappo_pretrain",
    )
    candidates = []
    if os.path.isdir(expert_root):
        for run_name in os.listdir(expert_root):
            run_dir = os.path.join(expert_root, run_name)
            for ckpt_name in ("best_agent.pt", "latest_agent.pt"):
                path = os.path.join(run_dir, ckpt_name)
                if os.path.exists(path):
                    candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No toy_cnn PlaceCucumber expert checkpoint found. "
            "Set --expert-agent-path or EXPERT_AGENT_PATH to a trained toy_cnn MAPPO checkpoint."
        )
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates


def load_toy_cnn_expert_state(args, expert: ToyCNNMAPPOAgent, expected_agent_names: Sequence[str]) -> str:
    candidates = resolve_toy_cnn_expert_candidates(args)
    errors = []
    for path in candidates:
        state_dict = torch.load(path, map_location="cpu")
        try:
            expert.load_state_dict(state_dict)
            return path
        except RuntimeError as exc:
            ckpt_agents = _checkpoint_agent_names(state_dict)
            errors.append(
                f"{path}: checkpoint_agents={ckpt_agents}, "
                f"expected_agents={list(expected_agent_names)}, error={str(exc).splitlines()[0]}"
            )
            if args.expert_agent_path:
                raise RuntimeError(
                    "Incompatible toy_cnn expert checkpoint for PlaceCucumber collection.\n"
                    + "\n".join(errors)
                    + "\nUse a PlaceCucumber-v1 toy_cnn checkpoint trained with the current env/obs keys."
                ) from exc

    raise RuntimeError(
        "No compatible toy_cnn PlaceCucumber expert checkpoint found.\n"
        + "\n".join(errors[-8:])
        + "\nSet EXPERT_AGENT_PATH to a compatible "
        "ckpt/PlaceCucumber-v1/ppo/panda_widowx_widowx/toy_cnn_place_cucumber_mappo_pretrain/<run>/best_agent.pt."
    )


def build_toy_cnn_expert(args, agent_names: Sequence[str], device: torch.device):
    env_kwargs = create_env_kwargs(args, render_mode="rgb_array", sim_backend=None)
    env_kwargs["shader_dir"] = "default"
    agent_obs_rules = build_toy_cnn_agent_obs_rules(agent_names)
    collate_fn = make_toy_cnn_collate_fn(device)
    toy_args = argparse.Namespace(
        task_name=args.task_name,
        seed=args.seed,
        normalize_state=args.expert_normalize_state,
    )
    infos = get_toy_cnn_agent_info(toy_args, env_kwargs, agent_obs_rules, collate_fn)
    expert = ToyCNNMAPPOAgent(
        agent_names=infos["agent_names"],
        state_dims=infos["state_dims"],
        global_state_dim=infos["global_state_dim"],
        action_dims=infos["action_dims"],
        camera_count=1,
        normalize_state=args.expert_normalize_state,
    ).to(device)
    expert_path = load_toy_cnn_expert_state(args, expert, infos["agent_names"])
    expert.eval()
    print(f"[Collect] loaded toy_cnn expert: {expert_path}")
    return expert, agent_obs_rules, collate_fn


def flatten_expert_actions(action_dict: Dict[str, np.ndarray], agent_names: Sequence[str]) -> np.ndarray:
    flat_parts = []
    for name in agent_names:
        value = action_dict[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        flat_parts.append(np.asarray(value, dtype=np.float32))
    return np.concatenate(flat_parts, axis=-1)


def remove_trajectory_files(traj_path: str) -> None:
    for stale_path in (traj_path, traj_path.replace(".h5", ".json")):
        if os.path.exists(stale_path):
            os.remove(stale_path)


def validate_trajectory_dataset(traj_path: str, require_success: bool = True) -> tuple[bool, str]:
    json_path = traj_path.replace(".h5", ".json")
    if not os.path.exists(traj_path):
        return False, f"missing h5: {traj_path}"
    if not os.path.exists(json_path):
        return False, f"missing json: {json_path}"
    try:
        with open(json_path, "r") as f:
            meta = json.load(f)
        episodes = list(meta.get("episodes", []))
        if require_success:
            episodes = [episode for episode in episodes if bool(episode.get("success", False))]
        if not episodes:
            return False, "no successful episodes in json"
        first_episode_id = int(episodes[0]["episode_id"])
        with h5py.File(traj_path, "r") as h5_file:
            traj_key = f"traj_{first_episode_id}"
            if traj_key not in h5_file:
                return False, f"missing {traj_key} in h5"
            traj = h5_file[traj_key]
            for key in ("obs", "actions"):
                if key not in traj:
                    return False, f"missing {traj_key}/{key}"
            if "sensor_data" not in traj["obs"] or "extra" not in traj["obs"] or "agent" not in traj["obs"]:
                return False, f"incomplete obs tree in {traj_key}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "ok"


def collect_expert_trajectories(args, ckpt: Dict[str, str]) -> str:
    traj_path = args.trajectory_h5_path or ckpt["trajectory_h5"]
    json_path = traj_path.replace(".h5", ".json")
    if os.path.exists(traj_path) and os.path.exists(json_path) and not args.force_recollect_trajectories:
        valid, reason = validate_trajectory_dataset(traj_path, require_success=True)
        if valid:
            print(f"[Collect] reuse existing trajectories: {traj_path}")
            return traj_path
        print(f"[Collect] discard invalid trajectories: {traj_path} ({reason})")
        remove_trajectory_files(traj_path)

    output_dir = os.path.dirname(traj_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    env_kwargs = create_env_kwargs(args, render_mode="rgb_array", sim_backend="physx_cuda")
    if args.collect_shader_dir:
        env_kwargs["shader_dir"] = args.collect_shader_dir

    probe_env = gym.make(args.task_name, num_envs=1, **env_kwargs)
    agent_names = list(probe_env.unwrapped.agent.agents_dict.keys())
    probe_env.close()
    expert, agent_obs_rules, collate_fn = build_toy_cnn_expert(args, agent_names, device)

    base_env = gym.make(args.task_name, num_envs=args.num_collect_envs, **env_kwargs)
    flat_env = FlattenActionSpaceWrapper(base_env)
    record_env = RecordEpisode(
        flat_env,
        output_dir=output_dir,
        save_trajectory=True,
        trajectory_name=Path(traj_path).stem,
        save_video=False,
        record_reward=False,
        record_env_state=False,
        source_type="rl",
        source_desc="Demonstrations generated by rolling out a pretrained toy_cnn MAPPO policy",
    )
    env = ToyCNNObservationWrapper(record_env, agent_obs_rules=agent_obs_rules)

    obs, _ = env.reset(seed=args.seed)
    successful_episodes = 0
    total_finished_episodes = 0
    progress = tqdm(total=args.num_successful_trajectories, ascii=True, desc="Collect Expert")
    max_finished = args.max_collect_episodes

    while successful_episodes < args.num_successful_trajectories:
        with torch.no_grad():
            batch = collate_fn(obs)
            actions = expert.get_action(batch, deterministic=not args.expert_sample_actions)
            actions = {
                name: action.detach().cpu().numpy().astype(np.float32)
                for name, action in actions.items()
            }
            actions = flatten_expert_actions(actions, agent_names)

        obs, _, terminations, truncations, infos = env.step(actions)
        done = (terminations | truncations).detach().cpu().numpy().astype(bool)

        if done.any():
            done_env_idx = np.flatnonzero(done)
            step_success = infos["success"][done_env_idx].detach().cpu().numpy().astype(bool)
            successful_episodes += int(step_success.sum())
            total_finished_episodes += int(done_env_idx.size)
            progress.n = min(successful_episodes, args.num_successful_trajectories)
            progress.set_postfix(
                finished=total_finished_episodes,
                success=successful_episodes,
                refresh=False,
            )
            progress.refresh()
            if max_finished is not None and total_finished_episodes >= max_finished:
                break
            obs, _ = env.reset(options={"env_idx": done_env_idx})

    progress.close()
    record_env.save_on_reset = False
    env.close()

    kept_successes = filter_successful_trajectories(
        traj_path,
        max_episodes=args.num_successful_trajectories,
    )
    print(
        f"[Collect] successful_episodes={successful_episodes}, "
        f"finished_episodes={total_finished_episodes}, "
        f"kept_success_episodes={kept_successes}, saved={traj_path}"
    )
    if kept_successes <= 0:
        remove_trajectory_files(traj_path)
        raise RuntimeError(
            f"Expert collection produced no successful trajectories. "
            f"expert={args.expert_agent_path or '<auto>'}, output={traj_path}"
        )
    valid, reason = validate_trajectory_dataset(traj_path, require_success=True)
    if not valid:
        remove_trajectory_files(traj_path)
        raise RuntimeError(f"Collected trajectory dataset is invalid: {traj_path} ({reason})")
    return traj_path


def build_agent(args, infos: Dict[str, Any], device: torch.device):
    agent = MixedTinyVLAAdapterSmolVLAEfficientVLAAgent(
        **infos,
        model_dir=args.model_dir,
        normalize_state=args.normalize_state,
        freeze_vla_backbone=args.freeze_vla_backbone,
        critic_hidden_dim=args.critic_hidden_dim,
        attention_implementation=args.attn_implementation,
        image_size=args.image_size,
        use_vla_lora=args.use_vla_lora,
        use_vision_lora=args.use_vision_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_vision_backbone=args.train_vision_backbone,
        vision_token_pool_size=args.vision_token_pool_size,
        policy_mode=args.policy_mode,
        tiny_hidden_dim=args.tiny_hidden_dim,
        tiny_vision_layers=args.tiny_vision_layers,
        tiny_decoder_layers=args.tiny_decoder_layers,
        tiny_attention_heads=args.tiny_attention_heads,
        tiny_patch_size=args.tiny_patch_size,
        tiny_ffn_mult=args.tiny_ffn_mult,
        tiny_num_action_bins=args.tiny_num_action_bins,
        tiny_prompt_length=args.tiny_prompt_length,
        smolvla_hidden_dim=args.smolvla_hidden_dim,
        smolvla_vision_layers=args.smolvla_vision_layers,
        smolvla_attention_heads=args.smolvla_attention_heads,
        smolvla_patch_size=args.smolvla_patch_size,
        smolvla_ffn_mult=args.smolvla_ffn_mult,
        efficientvla_hidden_dim=args.efficientvla_hidden_dim,
        efficientvla_vision_layers=args.efficientvla_vision_layers,
        efficientvla_decoder_layers=args.efficientvla_decoder_layers,
        efficientvla_attention_heads=args.efficientvla_attention_heads,
        efficientvla_patch_size=args.efficientvla_patch_size,
        efficientvla_ffn_mult=args.efficientvla_ffn_mult,
    )
    return agent.to(device)


def build_sft_optimizer(args, agent: MixedTinyVLAAdapterSmolVLAEfficientVLAAgent) -> torch.optim.Optimizer:
    param_groups = [
        {
            "params": [p for p in agent.vla_actor.vla.parameters() if p.requires_grad],
            "lr": args.backbone_learning_rate,
            "group_name": "vla_adapter_backbone",
        },
        {
            "params": [p for p in agent.vla_actor.state_projector.parameters() if p.requires_grad],
            "lr": args.state_learning_rate,
            "group_name": "vla_adapter_state_projector",
        },
        {
            "params": [p for p in agent.vla_actor.context_projector.parameters() if p.requires_grad]
            + [p for p in agent.vla_actor.actor_head.parameters() if p.requires_grad],
            "lr": args.head_learning_rate,
            "group_name": "vla_adapter_heads",
        },
        {
            "params": [p for p in agent.smolvla_actor.vla.parameters() if p.requires_grad],
            "lr": args.backbone_learning_rate,
            "group_name": "smolvla_backbone",
        },
        {
            "params": [p for p in agent.smolvla_actor.state_projector.parameters() if p.requires_grad],
            "lr": args.state_learning_rate,
            "group_name": "smolvla_state_projector",
        },
        {
            "params": [p for p in agent.smolvla_actor.context_projector.parameters() if p.requires_grad]
            + [p for p in agent.smolvla_actor.action_mean_head.parameters() if p.requires_grad]
            + [p for p in agent.smolvla_actor.log_std_head.parameters() if p.requires_grad],
            "lr": args.head_learning_rate,
            "group_name": "smolvla_heads",
        },
        {
            "params": [p for p in agent.efficientvla_actor.vla.parameters() if p.requires_grad],
            "lr": args.backbone_learning_rate,
            "group_name": "efficientvla_backbone",
        },
        {
            "params": [p for p in agent.efficientvla_actor.state_projector.parameters() if p.requires_grad],
            "lr": args.state_learning_rate,
            "group_name": "efficientvla_state_projector",
        },
        {
            "params": [p for p in agent.efficientvla_actor.context_projector.parameters() if p.requires_grad]
            + [p for p in agent.efficientvla_actor.action_mean_head.parameters() if p.requires_grad]
            + [p for p in agent.efficientvla_actor.log_std_head.parameters() if p.requires_grad],
            "lr": args.head_learning_rate,
            "group_name": "efficientvla_heads",
        },
    ]
    param_groups = [group for group in param_groups if group["params"]]
    return torch.optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)


def resolve_trajectory_path(args, ckpt: Dict[str, str]) -> str:
    path = args.trajectory_h5_path or ckpt["trajectory_h5"]
    json_path = path.replace(".h5", ".json")
    if os.path.exists(path) and os.path.exists(json_path):
        return path
    if args.collect_only:
        raise RuntimeError(
            "PlaceCucumber BC does not collect expert trajectories yet. "
            "Please generate successful toy_cnn trajectories first and pass --trajectory-h5-path."
        )
    raise FileNotFoundError(
        f"Missing trajectory dataset: {path} and/or {json_path}. "
        "Pass --trajectory-h5-path to an existing successful trajectory h5."
    )


def filter_successful_trajectories(traj_path: str, max_episodes: int | None = None) -> int:
    json_path = traj_path.replace(".h5", ".json")
    with open(json_path, "r") as f:
        meta = json.load(f)

    successful_episodes = [episode for episode in meta["episodes"] if bool(episode.get("success", False))]
    if max_episodes is not None:
        successful_episodes = successful_episodes[:max_episodes]

    tmp_h5_path = f"{traj_path}.tmp"
    tmp_json_path = f"{json_path}.tmp"
    kept_episodes = []
    with h5py.File(traj_path, "r") as src_h5, h5py.File(tmp_h5_path, "w") as dst_h5:
        for new_episode_id, episode in enumerate(successful_episodes):
            src_h5.copy(f"traj_{episode['episode_id']}", dst_h5, name=f"traj_{new_episode_id}")
            new_episode = dict(episode)
            new_episode["episode_id"] = new_episode_id
            kept_episodes.append(new_episode)

    new_meta = dict(meta)
    new_meta["episodes"] = kept_episodes
    with open(tmp_json_path, "w") as f:
        json.dump(new_meta, f, indent=2)
    shutil.move(tmp_h5_path, traj_path)
    shutil.move(tmp_json_path, json_path)
    return len(kept_episodes)


def load_h5_item(node: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, h5py.Dataset):
        return node[()]
    return {key: load_h5_item(node[key]) for key in node.keys()}


def _slice_h5_tree(node: h5py.Group | h5py.Dataset, horizon: int) -> Any:
    if isinstance(node, h5py.Dataset):
        return node[:horizon]
    return {key: _slice_h5_tree(node[key], horizon) for key in node.keys()}


def _load_raw_obs(traj: h5py.Group, horizon: int, agent_names: Sequence[str]) -> Dict[str, Any]:
    raw_obs = {
        "sensor_data": {
            "base_camera": {
                "rgb": traj["obs"]["sensor_data"]["base_camera"]["rgb"][:horizon],
            }
        },
        "agent": {},
        "extra": _slice_h5_tree(traj["obs"]["extra"], horizon),
    }
    for name in agent_names:
        raw_obs["agent"][name] = {
            "qpos": traj["obs"]["agent"][name]["qpos"][:horizon],
            "qvel": traj["obs"]["agent"][name]["qvel"][:horizon],
        }
    return raw_obs


def _load_actions(traj: h5py.Group, horizon: int, agent_names: Sequence[str]) -> Dict[str, np.ndarray]:
    action_node = traj["actions"]
    if isinstance(action_node, h5py.Dataset):
        flat_actions = action_node[:horizon]
        if flat_actions.shape[-1] % len(agent_names) != 0:
            raise ValueError(
                f"Flat action dim {flat_actions.shape[-1]} is not divisible by num_agents={len(agent_names)}"
            )
        split_dim = flat_actions.shape[-1] // len(agent_names)
        return {
            name: flat_actions[:, idx * split_dim : (idx + 1) * split_dim]
            for idx, name in enumerate(agent_names)
        }
    actions = load_h5_item(action_node)
    return {name: np.asarray(actions[name][:horizon], dtype=np.float32) for name in agent_names}


def load_sft_dataset_from_trajectories(
    args,
    ckpt: Dict[str, str],
    trajectory_h5_path: str,
    agent_names: Sequence[str],
) -> Dict[str, Any]:
    cache_path = args.dataset_cache_path or ckpt["dataset_cache"]
    if os.path.exists(cache_path) and not args.force_rebuild_dataset_cache:
        print(f"[Dataset] loading cache {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    json_path = trajectory_h5_path.replace(".h5", ".json")
    with open(json_path, "r") as f:
        meta = json.load(f)
    success_episodes = [episode for episode in meta["episodes"] if bool(episode.get("success", False))]
    success_episodes = success_episodes[: args.num_successful_trajectories]
    if not success_episodes:
        raise RuntimeError(f"No successful episodes found in {json_path}")

    rgbs = []
    global_states = []
    trajectory_lengths = []
    per_agent_states = {name: [] for name in agent_names}
    per_agent_actions = {name: [] for name in agent_names}

    with h5py.File(trajectory_h5_path, "r") as h5_file:
        for episode in tqdm(success_episodes, ascii=True, desc="Build SFT Dataset"):
            traj = h5_file[f"traj_{episode['episode_id']}"]
            horizon = int(episode["elapsed_steps"])
            raw_obs = _load_raw_obs(traj, horizon, agent_names)
            actions = _load_actions(traj, horizon, agent_names)
            parsed = build_batch_from_obs(raw_obs, list(agent_names))

            rgbs.append(torch.as_tensor(parsed["rgb"]).to(dtype=torch.uint8, device="cpu"))
            global_states.append(torch.as_tensor(parsed["global_state"]).to(dtype=torch.float32, device="cpu"))
            trajectory_lengths.append(horizon)
            for name in agent_names:
                per_agent_states[name].append(
                    torch.as_tensor(parsed[f"agent_states_{name}"]).to(dtype=torch.float32, device="cpu")
                )
                per_agent_actions[name].append(
                    torch.as_tensor(actions[name]).to(dtype=torch.float32, device="cpu")
                )

    dataset = {
        "rgb": torch.cat(rgbs, dim=0),
        "global_state": torch.cat(global_states, dim=0),
        "trajectory_lengths": trajectory_lengths,
        "num_samples": int(sum(trajectory_lengths)),
        "num_trajectories": len(trajectory_lengths),
        "agent_names": list(agent_names),
        "trajectory_h5_path": trajectory_h5_path,
    }
    for name in agent_names:
        dataset[f"agent_states_{name}"] = torch.cat(per_agent_states[name], dim=0)
        dataset[f"actions_{name}"] = torch.cat(per_agent_actions[name], dim=0)

    save_checkpoint(cache_path, dataset)
    dump_json(
        f"{cache_path}.json",
        {
            "num_samples": dataset["num_samples"],
            "num_trajectories": dataset["num_trajectories"],
            "trajectory_lengths": [int(x) for x in dataset["trajectory_lengths"]],
            "trajectory_h5_path": trajectory_h5_path,
        },
    )
    print(f"[Dataset] cached to {cache_path}")
    return dataset


def fit_state_stats_from_dataset(agent, dataset: Dict[str, Any], chunk_size: int) -> None:
    if agent.actor_state_rms is None and agent.critic_state_rms is None:
        return
    agent.unfreeze_state_stats()
    with torch.no_grad():
        total = int(dataset["num_samples"])
        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            if agent.actor_state_rms is not None:
                for name in agent.agent_names:
                    agent.actor_state_rms[name].update(
                        dataset[f"agent_states_{name}"][start:end].to(agent.device, dtype=torch.float32)
                    )
            if agent.critic_state_rms is not None:
                agent.critic_state_rms.update(
                    dataset["global_state"][start:end].to(agent.device, dtype=torch.float32)
                )
    agent.freeze_state_stats()


def sample_batch(dataset: Dict[str, Any], indices: np.ndarray, agent_names: Sequence[str], device: torch.device):
    torch_indices = torch.as_tensor(indices, dtype=torch.long)
    batch = {
        "rgb": dataset["rgb"][torch_indices].to(device=device),
        "global_state": dataset["global_state"][torch_indices].to(device=device, dtype=torch.float32),
    }
    actions = {}
    for name in agent_names:
        batch[f"agent_states_{name}"] = dataset[f"agent_states_{name}"][torch_indices].to(
            device=device,
            dtype=torch.float32,
        )
        actions[name] = dataset[f"actions_{name}"][torch_indices].to(device=device, dtype=torch.float32)
    return batch, actions


def maybe_run_eval(args, ckpt: Dict[str, str], agent, agent_names: Sequence[str], train_iter: int, writer):
    env_kwargs = create_env_kwargs(args, render_mode="rgb_array", sim_backend=None)
    eval_envs = make_eval_envs(
        env_id=args.task_name,
        num_envs=args.num_eval_envs,
        sim_backend="gpu",
        env_kwargs=env_kwargs,
        video_dir=f"{ckpt['video_dir']}_train",
    )
    agent.eval()
    metrics = evaluate(
        n=args.eval_episodes,
        sample_fn=make_sample_fn(agent_names, agent, deterministic=True),
        eval_envs=eval_envs,
    )
    eval_envs.close()
    payload = {key: float(value.mean()) for key, value in metrics.items()}
    print(f"[Eval] iter={train_iter} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(payload.items())))
    if writer is not None:
        for key, value in payload.items():
            writer.add_scalar(f"eval/{key}", value, train_iter)
    return payload


def train_sft(args) -> None:
    device = get_device()
    ckpt = resolve_ckpt_dir(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = not args.ignore_torch_deterministic

    traj_path = args.trajectory_h5_path or ckpt["trajectory_h5"]
    traj_json_path = traj_path.replace(".h5", ".json")
    existing_valid = False
    if os.path.exists(traj_path) and os.path.exists(traj_json_path):
        existing_valid, invalid_reason = validate_trajectory_dataset(traj_path, require_success=True)
        if not existing_valid:
            print(f"[Dataset] discard invalid trajectories: {traj_path} ({invalid_reason})")
            remove_trajectory_files(traj_path)
    if args.collect_only or args.force_recollect_trajectories or not (
        existing_valid
    ):
        trajectory_h5_path = collect_expert_trajectories(args, ckpt)
    else:
        trajectory_h5_path = resolve_trajectory_path(args, ckpt)
    if args.collect_only:
        print(f"[Dataset] collected trajectories: {trajectory_h5_path}")
        return

    infos = get_agent_info(args)
    agent_names = infos["agent_names"]
    dataset = load_sft_dataset_from_trajectories(args, ckpt, trajectory_h5_path, agent_names)
    agent = build_agent(args, infos, device)
    total_params = sum(parameter.numel() for parameter in agent.parameters())
    trainable_params = sum(parameter.numel() for parameter in agent.parameters() if parameter.requires_grad)
    print(
        f"[Model] backbone={args.model_backbone} total_params={total_params / 1e6:.2f}M "
        f"trainable_params={trainable_params / 1e6:.2f}M"
    )

    optimizer = build_sft_optimizer(args, agent)
    if args.refit_state_stats or not os.path.exists(ckpt["latest_agent"]):
        fit_state_stats_from_dataset(agent, dataset, args.state_stats_batch_size)

    best_score = -1.0
    global_updates = 0
    processed_samples = 0
    metrics_log = load_json(ckpt["metrics"]) if os.path.exists(ckpt["metrics"]) else []

    if os.path.exists(ckpt["latest_agent"]):
        print(f"[Train] resume agent from {ckpt['latest_agent']}")
        agent.load_checkpoint_state_dict(torch.load(ckpt["latest_agent"], map_location="cpu"))
        agent.to(device)
    if os.path.exists(ckpt["latest_opt"]):
        latest_opt = torch.load(ckpt["latest_opt"], map_location="cpu")
        try:
            optimizer.load_state_dict(latest_opt["opt"])
        except ValueError as exc:
            print(f"[Train] skip optimizer state restore because parameter groups changed: {exc}")
        best_score = float(latest_opt.get("best_score", best_score))
        global_updates = int(latest_opt.get("iter", latest_opt.get("updates", 0)))
        processed_samples = int(latest_opt.get("processed_samples", global_updates * args.batch_size))

    writer = SummaryWriter(ckpt["log_dir"])
    total_samples = int(dataset["num_samples"])
    batches_per_pass = max(1, math.ceil(total_samples / args.batch_size))
    total_iters = args.sft_total_iters
    if total_iters is None:
        total_iters = args.sft_epochs * batches_per_pass
        print(
            f"[Train] --sft-total-iters not set; fallback to "
            f"sft_epochs({args.sft_epochs}) * batches_per_pass({batches_per_pass}) = {total_iters}"
        )
    total_iters = int(total_iters)
    if global_updates >= total_iters:
        print(f"[Train] already reached target iterations: {global_updates}/{total_iters}")
        writer.close()
        return

    pbar = tqdm(total=total_iters, initial=global_updates, ascii=True, desc="SFT Iters")
    train_start = time.time()
    shuffled = np.arange(total_samples)
    np.random.shuffle(shuffled)
    cursor = 0
    dataset_pass = 0
    window_loss = 0.0
    window_total_loss = 0.0
    window_entropy = 0.0
    window_entropy_per_dim = 0.0
    window_entropy_floor_loss = 0.0
    window_steps = 0

    while global_updates < total_iters:
        agent.train()
        if cursor >= total_samples:
            np.random.shuffle(shuffled)
            cursor = 0
            dataset_pass += 1

        end = min(cursor + args.batch_size, total_samples)
        mb_inds = shuffled[cursor:end]
        cursor = end
        batch, actions_input = sample_batch(dataset, mb_inds, agent_names, device)

        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = args.use_amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            _, log_probs, entropies, _ = agent.get_action_and_value(batch, actions_input=actions_input)
            mean_log_prob = torch.stack([log_probs[name] for name in agent_names], dim=0).mean()
            mean_entropy = torch.stack([entropies[name] for name in agent_names], dim=0).mean()
            entropy_per_dim = mean_entropy / max(1, int(agent.action_dim))
            bc_loss = (-mean_log_prob) / agent.action_dim
            entropy_floor_loss = torch.zeros_like(bc_loss)
            if args.min_entropy_per_dim is not None and args.min_entropy_per_dim > 0:
                entropy_floor = torch.as_tensor(
                    args.min_entropy_per_dim,
                    device=entropy_per_dim.device,
                    dtype=entropy_per_dim.dtype,
                )
                entropy_floor_loss = torch.relu(entropy_floor - entropy_per_dim)
            loss = (
                bc_loss
                - args.entropy_coef * entropy_per_dim
                + args.entropy_floor_coef * entropy_floor_loss
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        global_updates += 1
        processed_samples += len(mb_inds)
        window_steps += 1
        window_loss += float(bc_loss.detach().item())
        window_total_loss += float(loss.detach().item())
        window_entropy += float(mean_entropy.detach().item())
        window_entropy_per_dim += float(entropy_per_dim.detach().item())
        window_entropy_floor_loss += float(entropy_floor_loss.detach().item())
        writer.add_scalar("train/bc_loss_step", float(bc_loss.detach().item()), global_updates)
        writer.add_scalar("train/total_loss_step", float(loss.detach().item()), global_updates)
        writer.add_scalar("train/entropy_step", float(mean_entropy.detach().item()), global_updates)
        writer.add_scalar("train/entropy_per_dim_step", float(entropy_per_dim.detach().item()), global_updates)
        writer.add_scalar(
            "train/entropy_floor_loss_step",
            float(entropy_floor_loss.detach().item()),
            global_updates,
        )
        pbar.set_postfix(
            loss=f"{window_loss / max(window_steps, 1):.4f}",
            ent_dim=f"{window_entropy_per_dim / max(window_steps, 1):.4f}",
        )
        pbar.update(1)

        should_log = global_updates % args.log_interval_iters == 0 or global_updates == total_iters
        should_eval = global_updates % args.eval_interval_iters == 0 or global_updates == total_iters
        eval_payload = {}
        score = None
        if should_eval:
            eval_payload = maybe_run_eval(args, ckpt, agent, agent_names, global_updates, writer)
            score = eval_payload.get("success_rate")
            if score is None:
                score = eval_payload.get("success_once")
            if score is None and eval_payload:
                score = next(iter(eval_payload.values()))
            if score is not None and score >= best_score:
                best_score = score
                save_checkpoint(ckpt["best_agent"], agent.checkpoint_state_dict())
                print(f"[Eval] new best model saved, score={score:.4f}")

        if should_log or should_eval:
            avg_loss = window_loss / max(window_steps, 1)
            avg_total_loss = window_total_loss / max(window_steps, 1)
            avg_entropy = window_entropy / max(window_steps, 1)
            avg_entropy_per_dim = window_entropy_per_dim / max(window_steps, 1)
            avg_entropy_floor_loss = window_entropy_floor_loss / max(window_steps, 1)
            samples_per_sec = processed_samples / max(time.time() - train_start, 1e-6)
            writer.add_scalar("train/bc_loss_iter_window", avg_loss, global_updates)
            writer.add_scalar("train/total_loss_iter_window", avg_total_loss, global_updates)
            writer.add_scalar("train/entropy_iter_window", avg_entropy, global_updates)
            writer.add_scalar("train/entropy_per_dim_iter_window", avg_entropy_per_dim, global_updates)
            writer.add_scalar("train/entropy_floor_loss_iter_window", avg_entropy_floor_loss, global_updates)
            writer.add_scalar("train/samples_per_sec", samples_per_sec, global_updates)
            save_checkpoint(ckpt["latest_agent"], agent.checkpoint_state_dict())
            save_checkpoint(
                ckpt["latest_opt"],
                {
                    "opt": optimizer.state_dict(),
                    "iter": global_updates,
                    "updates": global_updates,
                    "processed_samples": processed_samples,
                    "best_score": best_score,
                },
            )
            row = {
                "iter": global_updates,
                "dataset_pass": dataset_pass,
                "bc_loss": float(avg_loss),
                "total_loss": float(avg_total_loss),
                "entropy": float(avg_entropy),
                "entropy_per_dim": float(avg_entropy_per_dim),
                "entropy_floor_loss": float(avg_entropy_floor_loss),
                "samples_per_sec": float(samples_per_sec),
            }
            if score is not None:
                row["score"] = float(score)
            row.update({f"eval_{key}": float(value) for key, value in eval_payload.items()})
            metrics_log.append(row)
            dump_json(ckpt["metrics"], metrics_log)
            print(
                f"[Train] iter={global_updates}/{total_iters} pass={dataset_pass} "
                f"bc_loss={avg_loss:.6f} total_loss={avg_total_loss:.6f} "
                f"entropy={avg_entropy:.6f} entropy_per_dim={avg_entropy_per_dim:.6f} "
                f"entropy_floor_loss={avg_entropy_floor_loss:.6f} samples_per_sec={samples_per_sec:.2f}"
            )
            window_loss = 0.0
            window_total_loss = 0.0
            window_entropy = 0.0
            window_entropy_per_dim = 0.0
            window_entropy_floor_loss = 0.0
            window_steps = 0

    pbar.close()
    writer.close()


def evaluate_only(args) -> None:
    device = get_device()
    ckpt = resolve_ckpt_dir(args)
    infos = get_agent_info(args)
    agent_names = infos["agent_names"]
    agent = build_agent(args, infos, device)
    if args.eval_agent_dir is None:
        raise ValueError("Please provide --eval-agent-dir for evaluation mode")
    checkpoint_path = os.path.join(args.eval_agent_dir, "best_agent.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(args.eval_agent_dir, "latest_agent.pt")
    agent.load_checkpoint_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    agent.to(device)
    payload = maybe_run_eval(args, ckpt, agent, agent_names, train_iter=0, writer=None)
    dump_json(os.path.join(ckpt["root_dir"], "eval_metrics.json"), payload)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="PlaceCucumber-v1")
    parser.add_argument("--seed", type=int, default=1788)
    parser.add_argument("--save-dir", type=str, default="ckpt")
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--robot-name", type=str, default="panda_widowx_widowx")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument(
        "--model-backbone",
        type=str,
        default="mixed_tiny_vla_smolvla_efficientvla",
        choices=["mixed_tiny_vla_smolvla_efficientvla"],
    )
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--obs-mode", type=str, default="rgb+state_dict")
    parser.add_argument("--control-mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--reward-mode", type=str, default="normalized_dense")
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--ignore-torch-deterministic", action="store_true")

    parser.add_argument("--trajectory-h5-path", type=str, default=None)
    parser.add_argument("--dataset-cache-path", type=str, default=None)
    parser.add_argument("--force-rebuild-dataset-cache", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--force-recollect-trajectories", action="store_true")
    parser.add_argument("--expert-agent-path", type=str, default=None)
    parser.add_argument("--expert-sample-actions", action="store_true")
    parser.add_argument("--expert-normalize-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-collect-envs", type=int, default=16)
    parser.add_argument("--max-collect-episodes", type=int, default=None)
    parser.add_argument("--collect-shader-dir", type=str, default="minimal")
    parser.add_argument("--filter-successful-trajectories", action="store_true")
    parser.add_argument("--num-successful-trajectories", type=int, default=1024)

    parser.add_argument("--sft-total-iters", type=int, default=None)
    parser.add_argument("--sft-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-interval-iters", type=int, default=2000)
    parser.add_argument("--state-stats-batch-size", type=int, default=1024)
    parser.add_argument("--refit-state-stats", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--normalize-state", action="store_true")
    parser.add_argument("--freeze-vla-backbone", action="store_true")
    parser.add_argument("--use-vla-lora", action="store_true")
    parser.add_argument("--use-vision-lora", action="store_true")
    parser.add_argument("--train-vision-backbone", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--vision-token-pool-size", type=int, default=16)

    parser.add_argument("--tiny-hidden-dim", type=int, default=384)
    parser.add_argument("--tiny-vision-layers", type=int, default=4)
    parser.add_argument("--tiny-decoder-layers", type=int, default=4)
    parser.add_argument("--tiny-attention-heads", type=int, default=6)
    parser.add_argument("--tiny-patch-size", type=int, default=14)
    parser.add_argument("--tiny-ffn-mult", type=int, default=4)
    parser.add_argument("--tiny-num-action-bins", type=int, default=256)
    parser.add_argument("--tiny-prompt-length", type=int, default=24)
    parser.add_argument("--smolvla-hidden-dim", type=int, default=384)
    parser.add_argument("--smolvla-vision-layers", type=int, default=6)
    parser.add_argument("--smolvla-attention-heads", type=int, default=6)
    parser.add_argument("--smolvla-patch-size", type=int, default=14)
    parser.add_argument("--smolvla-ffn-mult", type=int, default=4)
    parser.add_argument("--efficientvla-hidden-dim", type=int, default=256)
    parser.add_argument("--efficientvla-vision-layers", type=int, default=4)
    parser.add_argument("--efficientvla-decoder-layers", type=int, default=1)
    parser.add_argument("--efficientvla-attention-heads", type=int, default=4)
    parser.add_argument("--efficientvla-patch-size", type=int, default=16)
    parser.add_argument("--efficientvla-ffn-mult", type=int, default=3)

    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--state-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--min-entropy-per-dim", type=float, default=None)
    parser.add_argument("--entropy-floor-coef", type=float, default=0.0)
    parser.add_argument("--critic-hidden-dim", type=int, default=512)
    parser.add_argument("--eval-interval-iters", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--num-eval-envs", type=int, default=16)
    parser.add_argument("--evaluate-mode", action="store_true")
    parser.add_argument("--eval-agent-dir", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--policy-mode", type=str, default="native", choices=["residual", "native"])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.filter_successful_trajectories:
        ckpt = resolve_ckpt_dir(args)
        trajectory_path = resolve_trajectory_path(args, ckpt)
        kept = filter_successful_trajectories(trajectory_path, max_episodes=args.num_successful_trajectories)
        print(f"[Dataset] kept {kept} successful trajectories in {trajectory_path}")
        return
    if args.evaluate_mode:
        evaluate_only(args)
        return
    train_sft(args)


if __name__ == "__main__":
    main()
