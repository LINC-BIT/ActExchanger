import argparse
import os
import random
import sys
import time
from typing import Dict, List

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

import envs.pass_object_five_robots  # noqa: F401
from train.marl.mappo.base import collect_rollout, mappo_update_on_policy, mappo_update_on_policy_ag
from train.reinforcement_learning.make_env import make_eval_envs
from train.reinforcement_learning.utils import compute_gae, get_stage, get_step_infos
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.mappo_feature_aggregator_pretrain import (
    build_clients,
    get_training_phase,
    iter_client_feature_aggregators,
    set_client_feature_aggregator_requires_grad,
    set_optimizer_phase_lrs,
    set_requires_grad,
)
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.baseline_adapters import (
    DICGStateAdapter,
    MATStateAdapter,
    PerAgentProjectInOutAdapter,
    TGCNetStateAdapter,
)
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.mixed_agent import (
    MixedFiveVLA100MAgent,
    build_mixed_mappo_optimizer,
)
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.model import (
    build_batch_from_obs,
)

mp.set_start_method("spawn", force=True)


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None


class FrozenMixedFiveVLABackboneAdapterAgent(MixedFiveVLA100MAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def enable_critic_training(self) -> None:
        for module in (self.critic_state_encoder, self.critic_visual_encoder, self.critic):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _adapter_residuals(self, state_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _adapt_actor_state_features(self, state_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        residuals = self._adapter_residuals(state_features)
        return {
            name: state_features[name] + residuals[name]
            for name in self.agent_names
        }


class FiveVLAMATVLABenchmarkAgent(FrozenMixedFiveVLABackboneAdapterAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mat_adapter = PerAgentProjectInOutAdapter(
            input_dims=[self.branch_hidden_dims[name] for name in self.agent_names],
            shared_dim=256,
            core=MATStateAdapter(hidden_dim=256, num_agents=len(self.agent_names)),
        )
        self.enable_critic_training()

    def _adapter_residuals(self, state_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.mat_adapter([state_features[name] for name in self.agent_names])
        return {name: output for name, output in zip(self.agent_names, outputs)}


class FiveVLADICGVLABenchmarkAgent(FrozenMixedFiveVLABackboneAdapterAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dicg_adapter = PerAgentProjectInOutAdapter(
            input_dims=[self.branch_hidden_dims[name] for name in self.agent_names],
            shared_dim=256,
            core=DICGStateAdapter(hidden_dim=256),
        )
        self.enable_critic_training()

    def _adapter_residuals(self, state_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.dicg_adapter([state_features[name] for name in self.agent_names])
        return {name: output for name, output in zip(self.agent_names, outputs)}


class FiveVLATGCNetVLABenchmarkAgent(FrozenMixedFiveVLABackboneAdapterAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tgcnet_adapter = PerAgentProjectInOutAdapter(
            input_dims=[self.branch_hidden_dims[name] for name in self.agent_names],
            shared_dim=256,
            core=TGCNetStateAdapter(hidden_dim=256, num_agents=len(self.agent_names)),
        )
        self.enable_critic_training()

    def _adapter_residuals(self, state_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.tgcnet_adapter([state_features[name] for name in self.agent_names])
        return {name: output for name, output in zip(self.agent_names, outputs)}


def build_vla_adapter_optimizer(args, agent: FrozenMixedFiveVLABackboneAdapterAgent) -> torch.optim.Optimizer:
    param_groups = []

    def append_group(parameters, lr, group_name):
        params = [parameter for parameter in parameters if parameter.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr, "group_name": group_name})

    adapter_modules = []
    if hasattr(agent, "mat_adapter"):
        adapter_modules.append(("mat_adapter", agent.mat_adapter))
    if hasattr(agent, "dicg_adapter"):
        adapter_modules.append(("dicg_adapter", agent.dicg_adapter))
    if hasattr(agent, "tgcnet_adapter"):
        adapter_modules.append(("tgcnet_adapter", agent.tgcnet_adapter))

    for group_name, module in adapter_modules:
        append_group(module.parameters(), args.head_learning_rate, group_name)
    append_group(agent.critic_state_encoder.parameters(), args.value_head_learning_rate, "critic_state_encoder")
    append_group(agent.critic_visual_encoder.parameters(), args.value_head_learning_rate, "critic_visual_encoder")
    append_group(agent.critic.parameters(), args.value_head_learning_rate, "critic")
    return torch.optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)


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
    info = {
        "agent_names": agent_names,
        "state_dim": batch[f"agent_states_{agent_names[0]}"].shape[-1],
        "state_dims": {name: batch[f"agent_states_{name}"].shape[-1] for name in agent_names},
        "global_state_dim": batch["global_state"].shape[-1],
        "action_dim": test_env.single_action_space[agent_names[0]].shape[0],
        "action_dims": {name: test_env.single_action_space[name].shape[0] for name in agent_names},
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


def create_agent_and_optimizer(args, infos):
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
        tiny_vla_use_decode_cache=args.tiny_vla_use_decode_cache,
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
        tinyvla_hidden_dim=args.tinyvla_hidden_dim,
        tinyvla_vision_layers=args.tinyvla_vision_layers,
        tinyvla_decoder_layers=args.tinyvla_decoder_layers,
        tinyvla_attention_heads=args.tinyvla_attention_heads,
        tinyvla_patch_size=args.tinyvla_patch_size,
        tinyvla_ffn_mult=args.tinyvla_ffn_mult,
        flower_hidden_dim=args.flower_hidden_dim,
        flower_vision_layers=args.flower_vision_layers,
        flower_decoder_layers=args.flower_decoder_layers,
        flower_attention_heads=args.flower_attention_heads,
        flower_patch_size=args.flower_patch_size,
        flower_ffn_mult=args.flower_ffn_mult,
    )

    if args.method in {"ours", "ours_online_wo_ag"}:
        agent = MixedFiveVLA100MAgent(**common_kwargs)
        return agent, None
    if args.method == "mat":
        agent = FiveVLAMATVLABenchmarkAgent(**common_kwargs)
        optimizer = build_vla_adapter_optimizer(args, agent)
        return agent, optimizer
    if args.method == "dicg":
        agent = FiveVLADICGVLABenchmarkAgent(**common_kwargs)
        optimizer = build_vla_adapter_optimizer(args, agent)
        return agent, optimizer
    if args.method == "tgcnet":
        agent = FiveVLATGCNetVLABenchmarkAgent(**common_kwargs)
        optimizer = build_vla_adapter_optimizer(args, agent)
        return agent, optimizer
    raise ValueError(f"Unsupported method={args.method}")


def cuda_sync(device):
    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


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

    agent, optimizer = create_agent_and_optimizer(args, infos)
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

    if args.method in {"ours", "ours_online_wo_ag"}:
        base_param_groups = []
        for group in build_mixed_mappo_optimizer(args, agent).param_groups:
            copied_group = {k: v for k, v in group.items() if k != "params"}
            copied_group["params"] = list(group["params"])
            base_param_groups.append(copied_group)

        for group in base_param_groups:
            group_name = group["group_name"]
            params = list(group["params"])
            if group_name in {"vla_adapter_heads", "smolvla_heads", "efficientvla_heads", "tinyvla_heads", "flower_heads"}:
                head_trainable_parameters.extend(params)
            elif group_name in {
                "vla_adapter_backbone",
                "vla_adapter_state_projector",
                "smolvla_backbone",
                "smolvla_state_projector",
                "efficientvla_backbone",
                "efficientvla_state_projector",
                "tinyvla_backbone",
                "tinyvla_state_projector",
                "flower_backbone",
                "flower_state_projector",
            }:
                encoder_trainable_parameters.extend(params)
            elif group_name in {"critic_state_encoder", "critic_visual_encoder", "critic"}:
                critic_trainable_parameters.extend(params)

        clients = build_clients(args, agent, infos, agent_names, device)
        client_infos = {name: clients[name].before_training_start(agent) for name in agent_names}
        for recv_name, client_recv in clients.items():
            for send_name, client_send_info in client_infos.items():
                if recv_name != send_name:
                    client_recv.add_feature_aggregator(send_name, client_send_info)

        if args.method == "ours":
            aggregator_param_groups = []
            for client in clients.values():
                feature_aggregators_parameters = client.get_feature_aggregators_parameters()
                for _, feature_aggregator_params in feature_aggregators_parameters.items():
                    params = list(feature_aggregator_params)
                    if not params:
                        continue
                    for parameter in params:
                        parameter.requires_grad = True
                    aggregator_param_groups.append(
                        {
                            "params": params,
                            "lr": args.feature_aggregator_learning_rate,
                            "eps": 1e-5,
                            "group_name": "feature_aggregator",
                        }
                    )
            optimizer = optim.AdamW(
                base_param_groups + aggregator_param_groups,
                eps=1e-5,
                weight_decay=args.weight_decay,
            )
        else:
            set_client_feature_aggregator_requires_grad(clients, False)
            optimizer = optim.AdamW(base_param_groups, eps=1e-5, weight_decay=args.weight_decay)
    elif optimizer is None:
        optimizer = build_mixed_mappo_optimizer(args, agent)

    agent, optimizer = accelerator.prepare(agent, optimizer)

    envs = build_train_envs(args)
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    if args.minibatch_size == 0:
        args.minibatch_size = step_infos["rollot_steps"] // args.num_minibatch // args.grad_accum_steps
    args.critic_only_update = False
    agent.freeze_state_stats()
    if clients is not None:
        set_optimizer_phase_lrs(args, optimizer, critic_only=False, train_actor_base=True)
        set_requires_grad(head_trainable_parameters, False)
        set_requires_grad(encoder_trainable_parameters, False)
        set_requires_grad(critic_trainable_parameters, True)

    global_steps = 0
    measured_steps = 0
    measured_time = 0.0
    rollout_sps = []
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
                clients=clients,
            )
            communication_snapshots = None
            if len(rollout) == 11:
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

            agent.eval()
            if clients is None:
                mappo_update_on_policy(
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
            elif args.method == "ours":
                all_aggregator_params = []
                has_active_aggregator = any(
                    feature_aggregator.remote_features is not None
                    for client in clients.values()
                    for feature_aggregator in iter_client_feature_aggregators(client)
                )
                for client in clients.values():
                    for feature_aggregator in iter_client_feature_aggregators(client):
                        all_aggregator_params.extend(list(feature_aggregator.module.parameters()))

                set_client_feature_aggregator_requires_grad(clients, True)
                if has_active_aggregator and all_aggregator_params:
                    mappo_update_on_policy_ag(
                        args,
                        agent,
                        optimizer,
                        data,
                        collate_fn,
                        accelerator,
                        phase["name"],
                        clients,
                        writer,
                        -1,
                    )
                else:
                    mappo_update_on_policy(
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
            else:
                mappo_update_on_policy(
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
                for client_name, client in clients.items():
                    _ = client.export_feature_and_action()
                ag_data_infos = {}
                for client_name, client in clients.items():
                    ag_data_infos[client_name] = client.export_feature_and_action()
                for recv_name, client_recv in clients.items():
                    for send_name, ag_data in ag_data_infos.items():
                        if recv_name != send_name:
                            client_recv.receive_feature_and_action(send_name, ag_data)

            agent.unfreeze_state_stats()
            agent.update_state_stats(obs_buf.reshape((-1,)), update_actor=False, update_critic=True)
            agent.update_state_stats(next_obs, update_actor=False, update_critic=True)
            agent.freeze_state_stats()

            cuda_sync(device)
            rollout_elapsed = time.perf_counter() - rollout_start_time
            steps_this_rollout = step_infos["rollot_steps"]
            global_steps += steps_this_rollout

            current_sps = steps_this_rollout / max(rollout_elapsed, 1e-6)
            phase = "warmup" if rollout_idx < args.warmup_rollouts else "measure"
            print(
                f"[Rollout] idx={rollout_idx + 1}/{num_rollouts} phase={phase} "
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="ours_online_wo_ag",
        choices=["ours", "ours_online_wo_ag", "mat", "dicg", "tgcnet"],
    )
    parser.add_argument("--task-name", type=str, default="PassObjectFiveRobots-v1")
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
    parser.add_argument("--actor-warmup-rollouts", type=int, default=0)
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
    parser.add_argument("--task-name-for-save", type=str, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--robot-name", type=str, default="panda_so100_widowx_xarm6_inspire")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--model-backbone", type=str, default="mixed_five_vla_100m")
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
    parser.add_argument("--tiny-hidden-dim", type=int, default=384)
    parser.add_argument("--tiny-vision-layers", type=int, default=4)
    parser.add_argument("--tiny-decoder-layers", type=int, default=4)
    parser.add_argument("--tiny-attention-heads", type=int, default=6)
    parser.add_argument("--tiny-patch-size", type=int, default=14)
    parser.add_argument("--tiny-ffn-mult", type=int, default=4)
    parser.add_argument("--tiny-num-action-bins", type=int, default=256)
    parser.add_argument("--tiny-prompt-length", type=int, default=24)
    parser.add_argument("--tiny-vla-use-decode-cache", action="store_true")
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
    parser.add_argument("--tinyvla-hidden-dim", type=int, default=320)
    parser.add_argument("--tinyvla-vision-layers", type=int, default=4)
    parser.add_argument("--tinyvla-decoder-layers", type=int, default=2)
    parser.add_argument("--tinyvla-attention-heads", type=int, default=4)
    parser.add_argument("--tinyvla-patch-size", type=int, default=16)
    parser.add_argument("--tinyvla-ffn-mult", type=int, default=3)
    parser.add_argument("--flower-hidden-dim", type=int, default=320)
    parser.add_argument("--flower-vision-layers", type=int, default=4)
    parser.add_argument("--flower-decoder-layers", type=int, default=2)
    parser.add_argument("--flower-attention-heads", type=int, default=4)
    parser.add_argument("--flower-patch-size", type=int, default=16)
    parser.add_argument("--flower-ffn-mult", type=int, default=3)
    parser.add_argument("--critic-hidden-dim", type=int, default=512)
    parser.add_argument("--phase1-end-step", type=int, default=500000)
    parser.add_argument("--phase2-end-step", type=int, default=1000000)
    parser.add_argument("--feature-selector-alpha", type=float, default=0.2)
    parser.add_argument("--feature-selector-topk-trajectories", type=int, default=4)
    parser.add_argument("--feature-selector-temporal-pool-steps", type=int, default=16)
    parser.add_argument(
        "--feature-selector-strategy",
        type=str,
        default="topk_return",
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
    parser.add_argument("--feature-aggregator-feature-gate-open-max", type=float, default=0.25)
    parser.add_argument("--feature-aggregator-action-gate-open-max", type=float, default=0.10)
    parser.add_argument("--feature-aggregator-q-ret-weight", type=float, default=0.85)
    parser.add_argument("--feature-aggregator-q-attn-weight", type=float, default=0.15)
    parser.add_argument("--feature-aggregator-remote-dropout-prob", type=float, default=0.0)
    parser.add_argument("--feature-aggregator-remote-noise-std", type=float, default=0.0)
    parser.add_argument("--feature-aggregator-remote-stale-shift-max", type=int, default=0)
    parser.add_argument("--gate-reg-coef", type=float, default=0.0)
    parser.add_argument("--gate-target-mean", type=float, default=0.6)
    parser.add_argument("--gate-std-coef", type=float, default=0.0)
    parser.add_argument("--feature-gate-reg-coef", type=float, default=None)
    parser.add_argument("--feature-gate-target-mean", type=float, default=None)
    parser.add_argument("--feature-gate-std-coef", type=float, default=None)
    parser.add_argument("--action-gate-reg-coef", type=float, default=None)
    parser.add_argument("--action-gate-target-mean", type=float, default=None)
    parser.add_argument("--action-gate-std-coef", type=float, default=None)
    parser.add_argument("--feature-gate-quality-coef", type=float, default=0.0)
    parser.add_argument("--action-gate-quality-coef", type=float, default=0.0)
    parser.add_argument("--feature-consistency-coef", type=float, default=0.0)
    parser.add_argument("--action-consistency-coef", type=float, default=0.0)
    parser.add_argument("--feature-attn-entropy-coef", type=float, default=0.0)
    parser.add_argument("--action-attn-entropy-coef", type=float, default=0.0)
    parser.add_argument("--feature-attn-diversity-coef", type=float, default=0.0)
    parser.add_argument("--action-attn-diversity-coef", type=float, default=0.0)
    parser.add_argument("--disable-ag-debug-histograms", action="store_true")
    parser.set_defaults(communication_replay=False)
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
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
