import argparse
import os
import random
import sys
import time

sys.path.append(os.getcwd())

from train.toy_cnn.multi_agents.two_robot_pick.gpu_auto_select import configure_cuda_visible_devices

configure_cuda_visible_devices()

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from accelerate import Accelerator
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import envs.two_robot_stack_cube_v1  # noqa: F401
from train.marl.comatrack.base import comatrack_update_on_policy
from train.marl.maple import collect_rollout as maple_collect_rollout, maple_update_on_policy
from train.marl.magrpo.base import magrpo_update_on_policy
from train.marl.maporl.base import maporl_update_on_policy
from train.marl.mappo.base import collect_rollout, mappo_update_on_policy, mappo_update_on_policy_ag
from train.marl.mpdf.base import mpdf_update_on_policy
from train.reinforcement_learning.make_env import make_eval_envs
from train.reinforcement_learning.utils import compute_gae, get_stage, get_step_infos
from train.vla_adapter_openvla.multi_agents.two_robot_stack.dicg_online_rl import (
    OpenVLADICGOnlineAgent,
    build_optimizer as build_dicg_optimizer,
)
from train.vla_adapter_openvla.multi_agents.two_robot_stack.maple_mixed_agent import (
    OpenVLAMapleOnlineAgent,
    build_optimizer as build_maple_optimizer,
)
from train.vla_adapter_openvla.multi_agents.two_robot_stack.mat_online_rl import (
    OpenVLAMATOnlineAgent,
    build_optimizer as build_mat_optimizer,
)
from train.vla_adapter_openvla.multi_agents.two_robot_stack.mappo_feature_aggregator_pretrain import (
    build_clients,
    get_training_phase,
    iter_client_feature_aggregators,
    set_client_feature_aggregator_requires_grad,
    set_optimizer_phase_lrs,
    set_requires_grad,
)
from train.vla_adapter_openvla.multi_agents.two_robot_stack.mixed_sft_agent import (
    CANONICAL_MIXED_MODEL_BACKBONE,
    MixedTinyVLAAdapterOpenVLASFTAgent,
    build_mixed_mappo_optimizer,
)
from train.vla_adapter_openvla.multi_agents.two_robot_stack.model import build_batch_from_obs
from train.vla_adapter_openvla.multi_agents.two_robot_stack.planner_model import (
    DEFAULT_SUBTASK_VOCAB,
    HighLevelSubtaskPlannerAgent,
)
from train.vla_adapter_openvla.multi_agents.two_robot_stack.planner_pretrain import (
    build_low_level_agent,
    build_optimizer as build_planner_optimizer,
    collect_planner_rollout,
)
from train.vla_adapter_openvla.multi_agents.two_robot_stack.tgcnet_online_rl import (
    OpenVLATGCNetOnlineAgent,
    build_optimizer as build_tgcnet_optimizer,
)

mp.set_start_method("spawn", force=True)


DEFAULT_INIT_AGENT_PATHS = {
    "ours_online_wo_ag": (
        "ckpt/TwoRobotStackCubeUR10e-v1_online_wo_ag/ppo/"
        "panda_ur10e_panda_gripper/vla_adapter_openvla_mappo/20260728-173446/latest_agent.pt"
    ),
    "mappo": (
        "ckpt/TwoRobotStackCubeUR10e-v1_online_baseline/ppo/"
        "panda_ur10e_panda_gripper/vla_adapter_openvla_mappo/20260728-051654/latest_agent.pt"
    ),
    "comatrack": (
        "ckpt/TwoRobotStackCubeUR10e-v1_online_baseline/comatrack/"
        "panda_ur10e_panda_gripper/vla_adapter_openvla_comatrack/20260728-121213/latest_agent.pt"
    ),
    "mat": (
        "ckpt/TwoRobotStackCubeUR10e-v1_online_baseline/mat/"
        "panda_ur10e_panda_gripper/vla_adapter_openvla_mat/20260728-051728/latest_agent.pt"
    ),
    "dicg": (
        "ckpt/TwoRobotStackCubeUR10e-v1_online_baseline/dicg/"
        "panda_ur10e_panda_gripper/vla_adapter_openvla_dicg/20260728-051740/latest_agent.pt"
    ),
    "tgcnet": (
        "ckpt/TwoRobotStackCubeUR10e-v1_online_baseline/tgcnet/"
        "panda_ur10e_panda_gripper/vla_adapter_openvla_tgcnet/20260728-051747/latest_agent.pt"
    ),
    "maple": (
        "ckpt/TwoRobotStackCubeUR10e-v1_online_baseline/maple/"
        "panda_ur10e_panda_gripper/vla_adapter_openvla_maple/20260728-051757/latest_agent.pt"
    ),
    "maporl": (
        "ckpt/TwoRobotStackCubeUR10e-v1_online_baseline/maporl/"
        "panda_ur10e_panda_gripper/vla_adapter_openvla_maporl_planner/20260728-121218/latest_agent.pt"
    ),
    "mpdf": (
        "ckpt/TwoRobotStackCubeUR10e-v1_online_baseline/mpdf/"
        "panda_ur10e_panda_gripper/vla_adapter_openvla_mpdf_planner/20260728-121222/latest_agent.pt"
    ),
    "magrpo": (
        "ckpt/TwoRobotStackCubeUR10e-v1_online_baseline/magrpo/"
        "panda_ur10e_panda_gripper/vla_adapter_openvla_magrpo_planner/20260728-121217/latest_agent.pt"
    ),
}
DEFAULT_LOW_LEVEL_AGENT_PATH = (
    "ckpt/TwoRobotStackCubeUR10e-v1/sft/"
    "panda_ur10e_panda_gripper/vla_adapter_openvla_sft/20260726-180924/latest_agent.pt"
)


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None


def make_collate_fn(agent_names):
    def collate_fn(obs):
        return build_batch_from_obs(obs, agent_names)

    return collate_fn


def get_agent_info(args):
    env_kwargs = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "rgb_array",
        "max_episode_steps": args.max_episode_steps,
    }
    test_env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend="gpu",
        env_kwargs=env_kwargs,
    )
    obs, _ = test_env.reset()
    agent_names = list(obs["agent"].keys())
    batch = build_batch_from_obs(obs, agent_names)
    action_dims = {
        agent_name: int(test_env.single_action_space[agent_name].shape[0])
        for agent_name in agent_names
    }
    info = {
        "agent_names": agent_names,
        "state_dim": batch[f"agent_states_{agent_names[0]}"].shape[-1],
        "global_state_dim": batch["global_state"].shape[-1],
        "action_dim": max(action_dims.values()),
        "action_dims": action_dims,
    }
    test_env.close()
    return info


def build_train_envs(args):
    env_kwargs = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "none",
        "max_episode_steps": args.max_episode_steps,
        "sim_backend": "physx_cuda",
    }
    envs = gym.make(args.task_name, num_envs=args.num_envs, **env_kwargs)
    return ManiSkillVectorEnv(
        envs,
        args.num_envs,
        ignore_terminations=args.ignore_partial_reset,
        record_metrics=True,
    )


def maybe_load_initial_agent(agent, init_agent_path):
    if not init_agent_path:
        return
    checkpoint = torch.load(init_agent_path, map_location="cpu")
    agent.load_checkpoint_state_dict(checkpoint)
    print(f"[Init] loaded agent checkpoint: {init_agent_path}")


def build_agent_and_runtime(args, infos):
    common_kwargs = dict(
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
    )

    if args.method in {"ours_online_wo_ag", "mappo", "comatrack"}:
        agent = MixedTinyVLAAdapterOpenVLASFTAgent(**common_kwargs)
        optimizer = build_mixed_mappo_optimizer(args, agent)
        update_fn = comatrack_update_on_policy if args.method == "comatrack" else mappo_update_on_policy
        return agent, optimizer, update_fn, collect_rollout, "mappo", False
    if args.method == "mat":
        agent = OpenVLAMATOnlineAgent(**common_kwargs)
        optimizer = build_mat_optimizer(args, agent)
        return agent, optimizer, mappo_update_on_policy, collect_rollout, "mappo", False
    if args.method == "dicg":
        agent = OpenVLADICGOnlineAgent(**common_kwargs)
        optimizer = build_dicg_optimizer(args, agent)
        return agent, optimizer, mappo_update_on_policy, collect_rollout, "mappo", False
    if args.method == "tgcnet":
        agent = OpenVLATGCNetOnlineAgent(**common_kwargs)
        optimizer = build_tgcnet_optimizer(args, agent)
        return agent, optimizer, mappo_update_on_policy, collect_rollout, "mappo", False
    if args.method == "maple":
        agent = OpenVLAMapleOnlineAgent(
            **common_kwargs,
            latent_layers=args.maple_latent_layers,
            future_horizon=args.maple_future_horizon,
        )
        optimizer = build_maple_optimizer(args, agent)
        return agent, optimizer, maple_update_on_policy, maple_collect_rollout, "maple", True
    if args.method in {"maporl", "mpdf", "magrpo"}:
        low_level_agent = build_low_level_agent(args, infos)
        agent = HighLevelSubtaskPlannerAgent(
            low_level_agent=low_level_agent,
            agent_names=infos["agent_names"],
            state_dim=infos["state_dim"],
            global_state_dim=infos["global_state_dim"],
            action_dim=infos["action_dim"],
            normalize_state=args.normalize_state,
            planner_hidden_dim=args.planner_hidden_dim,
            planner_layers=args.planner_layers,
            planner_subtasks=DEFAULT_SUBTASK_VOCAB,
            planner_text_mode=args.planner_text_mode,
            low_level_deterministic=args.planner_low_level_deterministic,
        )
        optimizer = build_planner_optimizer(args, agent)
        update_fn = {
            "maporl": maporl_update_on_policy,
            "mpdf": mpdf_update_on_policy,
            "magrpo": magrpo_update_on_policy,
        }[args.method]
        return agent, optimizer, update_fn, collect_planner_rollout, "mappo", True
    raise ValueError(f"Unsupported method={args.method}")


def cuda_sync(device):
    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def summarize_effective_tx(clients):
    if clients is None:
        return None

    summary = {
        "per_client": {},
        "total_bytes": 0,
        "matched_bytes": 0,
        "total_forwards": 0,
        "matched_forwards": 0,
        "effective_ratio": 0.0,
        "matched_forward_ratio": 0.0,
    }
    for client_name, client in clients.items():
        stats = client.debug_last_selected_payload_stats()
        if not isinstance(stats, dict):
            continue
        total_bytes = int(stats.get("total_bytes", 0))
        matched_bytes = int(stats.get("matched_bytes", 0))
        total_forwards = int(stats.get("total_forwards", 0))
        matched_forwards = int(stats.get("matched_forwards", 0))
        summary["per_client"][client_name] = {
            "total_bytes": total_bytes,
            "matched_bytes": matched_bytes,
            "effective_ratio": (matched_bytes / total_bytes) if total_bytes > 0 else 0.0,
            "total_forwards": total_forwards,
            "matched_forwards": matched_forwards,
            "matched_forward_ratio": (matched_forwards / total_forwards) if total_forwards > 0 else 0.0,
        }
        summary["total_bytes"] += total_bytes
        summary["matched_bytes"] += matched_bytes
        summary["total_forwards"] += total_forwards
        summary["matched_forwards"] += matched_forwards

    if summary["total_bytes"] > 0:
        summary["effective_ratio"] = summary["matched_bytes"] / summary["total_bytes"]
    if summary["total_forwards"] > 0:
        summary["matched_forward_ratio"] = summary["matched_forwards"] / summary["total_forwards"]
    return summary


def flatten_meta(meta):
    flat = {}
    for key, value in meta.items():
        if value is None:
            flat[key] = None
        elif isinstance(value, dict):
            flat[key] = {name: tensor.reshape(-1, *tensor.shape[2:]) for name, tensor in value.items()}
        elif torch.is_tensor(value):
            flat[key] = value.reshape(-1, *value.shape[2:]) if value.ndim >= 2 else value.reshape(-1)
        else:
            flat[key] = value
    return flat


def run_benchmark(args):
    torch.backends.cudnn.deterministic = not args.ignore_torch_deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator(mixed_precision="bf16" if args.use_amp else "no")
    device = accelerator.device
    writer = NullWriter()
    step_infos = get_step_infos(args)
    infos = get_agent_info(args)
    agent_names = infos["agent_names"]
    collate_fn = make_collate_fn(agent_names)

    agent, optimizer, update_fn, collect_rollout_fn, rollout_mode, train_mode_during_update = (
        build_agent_and_runtime(args, infos)
    )
    maybe_load_initial_agent(agent, args.init_agent_path)

    total_params = sum(parameter.numel() for parameter in agent.parameters())
    trainable_params = sum(parameter.numel() for parameter in agent.parameters() if parameter.requires_grad)
    print(
        f"[Model] method={args.method} total_params={total_params / 1e6:.2f}M "
        f"trainable_params={trainable_params / 1e6:.2f}M"
    )

    clients = None
    head_trainable_parameters = []
    encoder_trainable_parameters = []
    critic_trainable_parameters = []
    if args.method == "ours_online_wo_ag":
        clients = build_clients(args, agent, infos, agent_names, device)
        client_infos = {name: clients[name].before_training_start(agent) for name in agent_names}
        for recv_name, client_recv in clients.items():
            for send_name, client_send_info in client_infos.items():
                if recv_name != send_name:
                    client_recv.add_feature_aggregator(send_name, client_send_info)
        init_dir = os.path.dirname(args.init_agent_path)
        for name in agent_names:
            ag_path = os.path.join(init_dir, f"latest_ag_{name}.pt")
            if os.path.exists(ag_path):
                clients[name].load_feature_aggregators(ag_path)
                print(f"[Init] loaded feature aggregator: {ag_path}")
        set_client_feature_aggregator_requires_grad(clients, False)
        for group in optimizer.param_groups:
            group_name = group.get("group_name")
            params = list(group["params"])
            if group_name in {"vla_actor_heads", "smolvla_actor_heads"}:
                head_trainable_parameters.extend(params)
            elif group_name in {
                "vla_actor_vla",
                "vla_actor_state_projector",
                "smolvla_actor_vla",
                "smolvla_actor_state_projector",
            }:
                encoder_trainable_parameters.extend(params)
            elif group_name in {"critic_state_encoder", "critic_visual_encoder", "critic"}:
                critic_trainable_parameters.extend(params)

    envs = build_train_envs(args)
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    if args.minibatch_size == 0:
        args.minibatch_size = step_infos["rollot_steps"] // args.num_minibatch // args.grad_accum_steps
    agent, optimizer = accelerator.prepare(agent, optimizer)
    agent.freeze_state_stats()

    global_steps = 0
    measured_steps = 0
    measured_time = 0.0
    rollout_sps = []
    effective_tx_summaries = []
    num_rollouts = args.warmup_rollouts + args.measure_rollouts

    try:
        for rollout_idx in range(num_rollouts):
            if clients is not None:
                phase = get_training_phase(args, global_steps)
                set_optimizer_phase_lrs(
                    args,
                    optimizer,
                    critic_only=False,
                    train_actor_base=phase["train_encoder"],
                )
                set_requires_grad(head_trainable_parameters, phase["train_head"])
                set_requires_grad(encoder_trainable_parameters, phase["train_encoder"])
                set_requires_grad(critic_trainable_parameters, True)
                for client in clients.values():
                    client.train()

            agent.eval()
            cuda_sync(device)
            rollout_start_time = time.perf_counter()
            rollout_kwargs = dict(
                args=args,
                agent=agent,
                collate_fn=collate_fn,
                envs=envs,
                next_obs=next_obs,
                next_done=next_done,
                accelerator=accelerator,
                writer=writer,
                global_step=global_steps,
            )
            if clients is not None:
                rollout_kwargs["clients"] = clients
            rollout = collect_rollout_fn(**rollout_kwargs)

            communication_snapshots = None
            meta = None
            if rollout_mode == "maple":
                (
                    obs_buf,
                    action_buf,
                    logp_buf,
                    rew_buf,
                    done_buf,
                    val_buf,
                    final_val_buf,
                    next_obs,
                    next_done,
                    meta,
                ) = rollout
                old_token_logits_buf = None
            elif len(rollout) == 11:
                (
                    obs_buf,
                    action_buf,
                    logp_buf,
                    rew_buf,
                    done_buf,
                    val_buf,
                    final_val_buf,
                    next_obs,
                    next_done,
                    old_token_logits_buf,
                    communication_snapshots,
                ) = rollout
            elif len(rollout) == 10:
                (
                    obs_buf,
                    action_buf,
                    logp_buf,
                    rew_buf,
                    done_buf,
                    val_buf,
                    final_val_buf,
                    next_obs,
                    next_done,
                    old_token_logits_buf,
                ) = rollout
                communication_snapshots = None
            else:
                (
                    obs_buf,
                    action_buf,
                    logp_buf,
                    rew_buf,
                    done_buf,
                    val_buf,
                    final_val_buf,
                    next_obs,
                    next_done,
                ) = rollout
                old_token_logits_buf = None

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

            if rollout_mode == "maple":
                data = (
                    obs_buf.reshape((-1,)),
                    action_buf.reshape((-1,)),
                    logp_buf.reshape((-1,)),
                    adv_buf.reshape(-1),
                    ret_buf.reshape(-1),
                    val_buf.reshape(-1),
                    flatten_meta(meta),
                )
            else:
                data = (
                    obs_buf.reshape((-1,)),
                    action_buf.reshape((-1,)),
                    logp_buf.reshape((-1,)),
                    adv_buf.reshape(-1),
                    ret_buf.reshape(-1),
                    val_buf.reshape(-1),
                )
                if old_token_logits_buf is not None:
                    data = data + (old_token_logits_buf.reshape((-1,)),)
                if communication_snapshots is not None:
                    flat_communication_snapshots = []
                    num_envs = done_buf.shape[1]
                    for step_snapshot in communication_snapshots:
                        flat_communication_snapshots.extend([step_snapshot] * num_envs)
                    data = data + (flat_communication_snapshots,)

            if train_mode_during_update:
                agent.train()
            else:
                agent.eval()

            if args.method == "ours_online_wo_ag":
                update_fn(
                    args,
                    agent,
                    optimizer,
                    data,
                    collate_fn,
                    accelerator,
                    phase["name"],
                    writer,
                    -1,
                )
                ag_data_infos = {}
                for client_name, client in clients.items():
                    ag_data_infos[client_name] = client.export_feature_and_action()
                for recv_name, client_recv in clients.items():
                    for send_name, ag_data in ag_data_infos.items():
                        if recv_name != send_name:
                            client_recv.receive_feature_and_action(send_name, ag_data)
                effective_tx = summarize_effective_tx(clients)
                if effective_tx is not None:
                    effective_tx_summaries.append(effective_tx)
            else:
                update_fn(
                    args,
                    agent,
                    optimizer,
                    data,
                    collate_fn,
                    accelerator,
                    get_stage(global_steps, step_infos),
                    writer,
                    -1,
                )

            agent.unfreeze_state_stats()
            agent.update_state_stats(obs_buf.reshape((-1,)), update_actor=False, update_critic=True)
            agent.update_state_stats(next_obs, update_actor=False, update_critic=True)
            agent.freeze_state_stats()

            cuda_sync(device)
            rollout_elapsed = time.perf_counter() - rollout_start_time
            steps_this_rollout = step_infos["rollot_steps"]
            global_steps += steps_this_rollout
            current_sps = steps_this_rollout / max(rollout_elapsed, 1e-6)
            phase_name = "warmup" if rollout_idx < args.warmup_rollouts else "measure"
            print(
                f"[Rollout] idx={rollout_idx + 1}/{num_rollouts} phase={phase_name} "
                f"steps={steps_this_rollout} time={rollout_elapsed:.3f}s sps={current_sps:.3f}"
            )

            if rollout_idx >= args.warmup_rollouts:
                measured_steps += steps_this_rollout
                measured_time += rollout_elapsed
                rollout_sps.append(current_sps)
    finally:
        envs.close()

    mean_rollout_sps = float(np.mean(rollout_sps)) if rollout_sps else 0.0
    agg_sps = measured_steps / max(measured_time, 1e-6)
    print(
        "[BenchmarkResult] "
        f"method={args.method} measured_rollouts={args.measure_rollouts} "
        f"measured_steps={measured_steps} measured_time={measured_time:.3f}s "
        f"aggregate_sps={agg_sps:.3f} mean_rollout_sps={mean_rollout_sps:.3f}"
    )
    if effective_tx_summaries:
        latest = effective_tx_summaries[-1]
        print(
            "[EffectiveTxResult] "
            f"method={args.method} total_bytes={latest['total_bytes']} "
            f"matched_bytes={latest['matched_bytes']} "
            f"effective_ratio={latest['effective_ratio']:.6f} "
            f"total_forwards={latest['total_forwards']} "
            f"matched_forwards={latest['matched_forwards']} "
            f"matched_forward_ratio={latest['matched_forward_ratio']:.6f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="ours_online_wo_ag",
        choices=[
            "ours_online_wo_ag",
            "mappo",
            "comatrack",
            "mat",
            "dicg",
            "tgcnet",
            "maple",
            "maporl",
            "mpdf",
            "magrpo",
        ],
    )
    parser.add_argument("--task-name", type=str, default="TwoRobotStackCubeUR10e-v1")
    parser.add_argument("--seed", type=int, default=1788)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--warmup-rollouts", type=int, default=1)
    parser.add_argument("--measure-rollouts", type=int, default=3)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--num-minibatch", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--critic-warmup-rollouts", type=int, default=0)
    parser.add_argument("--save-interval-per-rollout", type=int, default=1000000)
    parser.add_argument("--total-steps", type=int, default=100000000)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--rollout-minibatch-size", type=int, default=0)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--not-normalize-adv", action="store_true")
    parser.add_argument("--finite-horizon-gae", action="store_true")
    parser.add_argument("--ignore-partial-reset", action="store_true")
    parser.add_argument("--ignore-torch-deterministic", action="store_true")
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-6)
    parser.add_argument("--head-learning-rate", type=float, default=3e-6)
    parser.add_argument("--state-learning-rate", type=float, default=3e-6)
    parser.add_argument("--feature-aggregator-learning-rate", type=float, default=3e-5)
    parser.add_argument("--value-head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.956)
    parser.add_argument("--gae-lambda", type=float, default=0.966)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--value-huber-delta", type=float, default=10.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.05)
    parser.add_argument("--aggregator-target-kl", type=float, default=2.0)
    parser.add_argument("--full-kl-coef", type=float, default=0.0)
    parser.add_argument("--log-full-kl", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--robot-name", type=str, default="panda_ur10e_panda_gripper")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--model-backbone", type=str, default=CANONICAL_MIXED_MODEL_BACKBONE)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--init-agent-path", type=str, default=None)
    parser.add_argument("--obs-mode", type=str, default="rgb+state_dict")
    parser.add_argument("--control-mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--reward-mode", type=str, default="normalized_dense")
    parser.add_argument("--freeze-vla-backbone", action="store_true")
    parser.add_argument("--use-vla-lora", action="store_true")
    parser.add_argument("--use-vision-lora", action="store_true")
    parser.add_argument("--train-vision-backbone", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--vision-token-pool-size", type=int, default=16)
    parser.add_argument("--tiny-hidden-dim", type=int, default=640)
    parser.add_argument("--tiny-vision-layers", type=int, default=7)
    parser.add_argument("--tiny-decoder-layers", type=int, default=8)
    parser.add_argument("--tiny-attention-heads", type=int, default=10)
    parser.add_argument("--tiny-patch-size", type=int, default=14)
    parser.add_argument("--tiny-ffn-mult", type=int, default=4)
    parser.add_argument("--tiny-num-action-bins", type=int, default=256)
    parser.add_argument("--tiny-prompt-length", type=int, default=24)
    parser.add_argument("--critic-hidden-dim", type=int, default=512)
    parser.add_argument("--phase1-end-step", type=int, default=500000)
    parser.add_argument("--phase2-end-step", type=int, default=1000000)
    parser.add_argument("--feature-selector-alpha", type=float, default=0.2)
    parser.add_argument("--feature-selector-topk-trajectories", type=int, default=4)
    parser.add_argument("--feature-selector-temporal-pool-steps", type=int, default=16)
    parser.add_argument(
        "--feature-selector-strategy",
        type=str,
        default="return_span",
        choices=["topk_return", "random", "return_span"],
    )
    parser.add_argument(
        "--eval-feature-selector-strategy",
        type=str,
        default=None,
        choices=["topk_return", "random", "return_span"],
    )
    parser.add_argument("--feature-aggregator-attention-num-heads", type=int, default=4)
    parser.add_argument(
        "--feature-aggregator-gate-type",
        type=str,
        default="two-layers",
        choices=["single-layer", "two-layers"],
    )
    parser.add_argument(
        "--feature-aggregator-gate-activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "silu", "tanh"],
    )
    parser.add_argument(
        "--feature-aggregator-norm-type",
        type=str,
        default="none",
        choices=["none", "layernorm"],
    )
    parser.add_argument("--feature-aggregator-feature-gate-open-max", type=float, default=0.20)
    parser.add_argument("--feature-aggregator-action-gate-open-max", type=float, default=0.06)
    parser.add_argument("--feature-aggregator-q-ret-weight", type=float, default=0.60)
    parser.add_argument("--feature-aggregator-q-attn-weight", type=float, default=0.40)
    parser.add_argument("--feature-aggregator-remote-dropout-prob", type=float, default=0.0)
    parser.add_argument("--feature-aggregator-remote-noise-std", type=float, default=0.0)
    parser.add_argument("--feature-aggregator-remote-stale-shift-max", type=int, default=0)
    parser.add_argument("--future-head-learning-rate", type=float, default=3e-6)
    parser.add_argument("--maple-latent-layers", type=int, default=2)
    parser.add_argument("--maple-future-horizon", type=int, default=1)
    parser.add_argument("--maple-future-state-coef", type=float, default=0.1)
    parser.add_argument("--maple-rl-bc-coef", type=float, default=0.0)
    parser.add_argument("--low-level-agent-path", type=str, default=None)
    parser.add_argument("--planner-hidden-dim", type=int, default=128)
    parser.add_argument("--planner-layers", type=int, default=1)
    parser.add_argument("--planner-text-mode", type=str, default="tiny_text", choices=["fixed", "tiny_text"])
    parser.add_argument("--planner-low-level-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--maporl-team-adv-coef", type=float, default=0.0)
    parser.add_argument("--maporl-kl-coef", type=float, default=0.0)
    parser.add_argument("--magrpo-eps", type=float, default=1e-5)
    parser.add_argument("--magrpo-gamma", type=float, default=1.0)
    parser.add_argument("--magrpo-group-size", type=int, default=0)
    parser.add_argument("--magrpo-kl-coef", type=float, default=0.0)
    parser.add_argument("--magrpo-ent-coef", type=float, default=0.0)
    parser.add_argument("--mpdf-eps", type=float, default=1e-5)
    parser.add_argument("--mpdf-group-size", type=int, default=0)
    parser.add_argument("--mpdf-rank-temperature", type=float, default=1.0)
    parser.add_argument("--mpdf-kl-coef", type=float, default=0.0)
    parser.add_argument("--mpdf-ent-coef", type=float, default=0.0)
    parser.add_argument("--gate-reg-coef", type=float, default=0.0)
    parser.add_argument("--gate-target-mean", type=float, default=0.1)
    parser.add_argument("--gate-std-coef", type=float, default=0.0)
    parser.add_argument("--feature-gate-reg-coef", type=float, default=None)
    parser.add_argument("--feature-gate-target-mean", type=float, default=0.1)
    parser.add_argument("--feature-gate-std-coef", type=float, default=None)
    parser.add_argument("--action-gate-reg-coef", type=float, default=None)
    parser.add_argument("--action-gate-target-mean", type=float, default=0.03)
    parser.add_argument("--action-gate-std-coef", type=float, default=None)
    parser.add_argument("--feature-gate-quality-coef", type=float, default=1.0)
    parser.add_argument("--action-gate-quality-coef", type=float, default=1.0)
    parser.add_argument("--feature-consistency-coef", type=float, default=0.5)
    parser.add_argument("--action-consistency-coef", type=float, default=1.5)
    parser.add_argument("--feature-attn-entropy-coef", type=float, default=1e-4)
    parser.add_argument("--action-attn-entropy-coef", type=float, default=1e-4)
    parser.add_argument("--feature-attn-diversity-coef", type=float, default=1e-4)
    parser.add_argument("--action-attn-diversity-coef", type=float, default=1e-4)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        default="native",
        choices=["residual", "native"],
    )
    args = parser.parse_args()
    if args.init_agent_path is None:
        args.init_agent_path = DEFAULT_INIT_AGENT_PATHS[args.method]
    if args.method in {"maporl", "mpdf", "magrpo"} and args.low_level_agent_path is None:
        args.low_level_agent_path = DEFAULT_LOW_LEVEL_AGENT_PATH
    if args.method == "maple":
        args.use_vla_lora = True
        args.lora_alpha = 16
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
