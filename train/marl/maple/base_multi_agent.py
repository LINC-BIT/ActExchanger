import inspect
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange

from train.marl.magrpo.base import build_group_ids, compute_group_relative_advantages
from train.marl.mappo.base import _compute_value_loss, get_trainable_optimizer_parameters
from train.reinforcement_learning.utils import DictArray


def _agent_supports_kwarg(agent, kwarg_name):
    try:
        signature = inspect.signature(agent.get_action_and_value)
    except (TypeError, ValueError, AttributeError):
        return False
    return kwarg_name in signature.parameters


def _ordered_agent_names(agent, rollout_action_dict=None):
    if hasattr(agent, "actor_heads"):
        return list(agent.actor_heads.keys())
    if hasattr(agent, "agent_names"):
        return list(agent.agent_names)
    if rollout_action_dict is not None:
        return list(rollout_action_dict.keys())
    return ["ego"]


def _to_device_tensor(value: Optional[Any], device: torch.device, *, dtype: Optional[torch.dtype] = None):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device)
    else:
        tensor = torch.as_tensor(value, device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _masked_weighted_mean(
    values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is not None:
        mask = mask.to(device=values.device, dtype=torch.bool)
        values = values[mask]
        if weight is not None:
            weight = weight[mask]
    if values.numel() == 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    if weight is None:
        return values.mean()
    weight = weight.to(device=values.device, dtype=values.dtype)
    denom = weight.sum().clamp_min(1e-8)
    return (values * weight).sum() / denom


def _apply_done_mask(x, mask):
    if torch.is_tensor(x):
        if x.shape[0] == mask.shape[0]:
            return x[mask]
        return x
    if isinstance(x, dict):
        return {k: _apply_done_mask(v, mask) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_apply_done_mask(v, mask) for v in x)
    return x


def _writer_add_scalar(writer, name: str, value: float, step: int) -> None:
    if writer is None:
        return
    writer.add_scalar(name, value, step)


def _normalize_rollout_output(agent, output):
    supports_action_bins = False
    if len(output) == 5:
        actions_dict, logp_dict, ent_dict, value, rollout_action_dict = output
        supports_action_bins = True
    elif len(output) == 4:
        actions_dict, logp_dict, ent_dict, value = output
        rollout_action_dict = actions_dict
    else:
        raise ValueError(f"Unexpected policy output length: {len(output)}")

    if not isinstance(actions_dict, Mapping):
        actions_dict = {"ego": actions_dict}
    if not isinstance(logp_dict, Mapping):
        logp_dict = {"ego": logp_dict}
    if not isinstance(ent_dict, Mapping):
        ent_dict = {"ego": ent_dict}
    if not isinstance(rollout_action_dict, Mapping):
        rollout_action_dict = {"ego": rollout_action_dict}

    if not hasattr(agent, "actor_heads"):
        agent.actor_heads = {name: None for name in actions_dict.keys()}

    return actions_dict, logp_dict, ent_dict, value, rollout_action_dict, supports_action_bins


def _policy_update_forward(agent, batch, stored_actions, return_aux: bool = False):
    kwargs = {}
    if _agent_supports_kwarg(agent, "action_bins_input"):
        kwargs["action_bins_input"] = {k: v for k, v in stored_actions.items()}
    else:
        kwargs["actions_input"] = {k: v for k, v in stored_actions.items()}
    if return_aux and _agent_supports_kwarg(agent, "return_aux"):
        kwargs["return_aux"] = True

    result = agent.get_action_and_value(batch, **kwargs)
    if return_aux and isinstance(result, tuple) and len(result) == 5 and isinstance(result[-1], Mapping):
        actions_dict, logp_dict, ent_dict, value, aux_outputs = result
        if not isinstance(actions_dict, Mapping):
            actions_dict = {"ego": actions_dict}
        if not isinstance(logp_dict, Mapping):
            logp_dict = {"ego": logp_dict}
        if not isinstance(ent_dict, Mapping):
            ent_dict = {"ego": ent_dict}
        return actions_dict, logp_dict, ent_dict, value, aux_outputs

    actions_dict, logp_dict, ent_dict, value, _, _ = _normalize_rollout_output(agent, result)
    return actions_dict, logp_dict, ent_dict, value, {}


def _infer_action_shape(single_action_space, agent_name: str):
    if hasattr(single_action_space, "spaces"):
        return single_action_space[agent_name].shape
    return single_action_space.shape


def _prepare_rollout_buffers(agent_names: Sequence[str], envs, T: int, N: int, device: torch.device):
    actions = {
        name: np.zeros((T, N) + _infer_action_shape(envs.single_action_space, name), dtype=np.float32)
        for name in agent_names
    }
    action_bins = {
        name: torch.zeros((T, N) + _infer_action_shape(envs.single_action_space, name), dtype=torch.long, device=device)
        for name in agent_names
    }
    logprobs = {name: torch.zeros((T, N), device=device) for name in agent_names}
    entropies = {name: torch.zeros((T, N), device=device) for name in agent_names}
    return (
        DictArray((T, N), None, data_dict=actions, device=device),
        DictArray((T, N), None, data_dict=action_bins, device=device),
        DictArray((T, N), None, data_dict=logprobs, device=device),
        DictArray((T, N), None, data_dict=entropies, device=device),
    )


def _unpack_training_data(data):
    if isinstance(data, Mapping):
        return (
            data["obs"],
            data["actions"],
            data["logprobs"],
            data["advantages"],
            data["returns"],
            data["values"],
            dict(data.get("meta", {})),
        )
    if len(data) == 6:
        return (*data, {})
    if len(data) == 7:
        return (*data[:6], dict(data[6]))
    raise ValueError(f"Unexpected MAPLE training tuple length: {len(data)}")


def _extract_future_state_target_from_batch(batch_like):
    if not isinstance(batch_like, Mapping):
        return None
    if "global_state" in batch_like and batch_like["global_state"] is not None:
        value = batch_like["global_state"]
    elif "states" in batch_like and batch_like["states"] is not None:
        value = batch_like["states"]
    else:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _reshape_meta_tensor(value, batch_size: int, device: torch.device, *, dtype=None):
    tensor = _to_device_tensor(value, device, dtype=dtype)
    if tensor is None:
        return None
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.ndim >= 2 and tensor.shape[0] * tensor.shape[1] == batch_size:
        return tensor.reshape(batch_size, *tensor.shape[2:])
    if tensor.numel() == batch_size:
        return tensor.reshape(batch_size)
    return tensor


def _meta_vector_flat(meta: Mapping[str, Any], key: str, mb_inds, batch_size: int, device, *, dtype=None):
    value = meta.get(key)
    if value is None:
        return None
    if isinstance(value, Mapping):
        if "ego" in value:
            value = value["ego"]
        else:
            raise TypeError(f"Expected vector-like meta field for '{key}', got mapping without 'ego'")
    tensor = _reshape_meta_tensor(value, batch_size, device, dtype=dtype)
    return tensor[mb_inds]


def _meta_agent_vector_flat(meta: Mapping[str, Any], key: str, agent_name: str, mb_inds, batch_size: int, device, *, dtype=None):
    container = meta.get(key)
    if container is None:
        return None
    if not isinstance(container, Mapping):
        raise TypeError(f"Expected mapping meta field for '{key}', got {type(container).__name__}")
    value = container.get(agent_name)
    if value is None:
        return None
    tensor = _reshape_meta_tensor(value, batch_size, device, dtype=dtype)
    return tensor[mb_inds]


def _compute_joint_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards, dtype=torch.float32)
    running = torch.zeros(rewards.shape[1], device=rewards.device, dtype=torch.float32)
    for step in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[step].to(torch.float32) + gamma * running * (1.0 - dones[step].to(torch.float32))
        returns[step] = running
    return returns


def _build_stepwise_group_meta(args, rewards: torch.Tensor, dones: torch.Tensor, device: torch.device):
    T, N = rewards.shape
    group_size = int(getattr(args, "maple_group_size", getattr(args, "magrpo_group_size", 0)))
    if group_size <= 0:
        return None, None, None, None
    groups_per_step = max(1, (N + group_size - 1) // group_size)
    local_group_ids = build_group_ids(N, group_size, device=device)
    group_ids = torch.stack([local_group_ids + step * groups_per_step for step in range(T)], dim=0)
    joint_returns = _compute_joint_returns(
        rewards=rewards,
        dones=dones,
        gamma=float(getattr(args, "maple_gamma", getattr(args, "magrpo_gamma", getattr(args, "gamma", 1.0)))),
    )
    shared_advantages = compute_group_relative_advantages(
        scores=joint_returns.reshape(-1),
        group_ids=group_ids.reshape(-1),
        eps=float(getattr(args, "maple_grpo_eps", getattr(args, "magrpo_eps", 1e-5))),
    ).reshape(T, N)
    turn_ids = torch.arange(T, device=device, dtype=torch.long).unsqueeze(1).expand(T, N)
    return group_ids, turn_ids, shared_advantages, joint_returns


def _future_state_aux_loss(args, aux_outputs: Mapping[str, Any], meta: Mapping[str, Any], mb_inds, batch_size: int, device):
    pred = aux_outputs.get("future_state")
    target = _meta_vector_flat(meta, "future_state_target", mb_inds, batch_size, device, dtype=torch.float32)
    if pred is None or target is None:
        return torch.zeros((), device=device)

    pred = torch.as_tensor(pred, device=device, dtype=torch.float32)
    if pred.shape != target.shape:
        target = target.view_as(pred)
    error = pred - target
    loss_type = getattr(args, "maple_future_state_loss", "huber")
    if loss_type == "mse":
        loss = error.pow(2)
    elif loss_type == "l1":
        loss = error.abs()
    else:
        delta = float(getattr(args, "maple_future_state_delta", 1.0))
        abs_error = error.abs()
        quadratic = torch.minimum(abs_error, torch.tensor(delta, device=device, dtype=abs_error.dtype))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + delta * linear
    mask = _meta_vector_flat(meta, "future_state_mask", mb_inds, batch_size, device, dtype=torch.float32)
    if mask is not None:
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(-1)
        denom = mask.sum().clamp_min(1.0)
        return (loss * mask).sum() / denom
    return loss.mean()


def collect_rollout(args, agent, collate_fn, envs, next_obs, next_done, accelerator, writer, global_step, clients=None):
    device = accelerator.device
    T = args.rollout_steps
    N = args.num_envs

    obs = DictArray((T, N), envs.single_observation_space, device=device)
    agent_names = _ordered_agent_names(agent)
    actions, action_bins, logprobs, entropies = _prepare_rollout_buffers(agent_names, envs, T, N, device)
    rewards = torch.zeros((T, N), device=device)
    dones = torch.zeros((T, N), device=device)
    values = torch.zeros((T, N), device=device)
    final_values = torch.zeros((T, N), device=device)

    initial_batch = collate_fn(next_obs)
    future_state_targets = None
    target0 = _extract_future_state_target_from_batch(initial_batch)
    if target0 is not None:
        future_state_targets = torch.zeros((T, N) + tuple(target0.shape[1:]), dtype=torch.float32, device=device)

    pbar = trange(T, ascii=True, desc="Collecting MAPLE Rollout")
    now_step = global_step

    for step in pbar:
        now_step += N
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            if hasattr(agent, "update_state_stats"):
                agent.update_state_stats(next_obs)
            batch = collate_fn(next_obs)
            actions_dict, logp_dict, ent_dict, value, rollout_action_dict, _ = _normalize_rollout_output(
                agent,
                agent.get_action_and_value(batch, return_action_bins=True),
            )

        values[step] = value.view(-1)
        for name in agent_names:
            action_bins[name][step] = torch.as_tensor(rollout_action_dict[name], device=device, dtype=torch.long)
            actions[name][step] = np.asarray(actions_dict[name], dtype=np.float32)
            logprobs[name][step] = logp_dict[name].detach().to(device)
            entropies[name][step] = ent_dict[name].detach().to(device)

        next_obs, reward, terminations, truncations, infos = envs.step(actions_dict)
        next_done = torch.logical_or(terminations, truncations).float()
        rewards[step] = reward.view(-1) * getattr(args, "reward_scale", 1.0)

        if future_state_targets is not None:
            next_batch = collate_fn(next_obs)
            target = _extract_future_state_target_from_batch(next_batch)
            if target is not None:
                future_state_targets[step] = _to_device_tensor(target, device, dtype=torch.float32)

        if clients is not None:
            for client_name, client in clients.items():
                action_mean = None
                if client_name in actions_dict:
                    action_mean = torch.as_tensor(actions_dict[client_name], device=device).detach()
                client.after_each_forward_during_rollout(
                    reward.view(-1).detach(),
                    next_done.detach(),
                    action_mean=action_mean,
                    success=infos.get("success", torch.zeros_like(next_done)).detach(),
                )

        if "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            for k, v in final_info["episode"].items():
                _writer_add_scalar(writer, f"rollout/{k}", v[done_mask].float().mean().item(), now_step)
            infos["final_observation"] = _apply_done_mask(infos["final_observation"], done_mask)
            with torch.no_grad():
                if hasattr(agent, "update_state_stats"):
                    agent.update_state_stats(infos["final_observation"])
                final_batch = collate_fn(infos["final_observation"])
                final_value = agent.get_value(final_batch).view(-1)
                final_values[step, torch.arange(N, device=device)[done_mask]] = final_value

    group_ids, turn_ids, shared_advantages, joint_returns = _build_stepwise_group_meta(
        args=args,
        rewards=rewards,
        dones=dones,
        device=device,
    )
    meta = {
        "future_state_target": future_state_targets,
        "using_action_bins": True,
        "joint_rewards": rewards,
        "joint_returns": joint_returns,
        "shared_advantages": shared_advantages,
        "group_ids": group_ids,
        "turn_ids": turn_ids,
        "sample_mask": torch.ones((T, N), dtype=torch.bool, device=device),
    }
    return (
        obs,
        action_bins,
        logprobs,
        rewards,
        dones,
        values,
        final_values,
        next_obs,
        next_done,
        meta,
    )


def maple_sft_update(args, agent, optimizer, data, collate_fn, accelerator, stage, writer):
    (
        b_obs,
        b_actions,
        _b_logprobs,
        _b_advantages,
        _b_returns,
        _b_values,
        meta,
    ) = _unpack_training_data(data)

    device = accelerator.device
    batch_size = len(b_obs)
    batch_indices = np.arange(batch_size)
    pbar = tqdm(total=args.update_epochs * max(1, batch_size // args.minibatch_size), desc=stage, ascii=True)
    stats = {
        "bc_loss": 0.0,
        "future_state_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
    }
    n_updates = 0
    accum_count = 0
    optimizer.zero_grad()

    future_state_coef = float(getattr(args, "maple_future_state_coef", 0.0))
    bc_coef = float(getattr(args, "maple_sft_bc_coef", 1.0))

    for _ in range(args.update_epochs):
        np.random.shuffle(batch_indices)
        for start in range(0, batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = batch_indices[start:end]
            batch = collate_fn(b_obs[mb_inds])
            _, new_logp, entropy, _new_value, aux_outputs = _policy_update_forward(
                agent,
                batch,
                {k: v[mb_inds] for k, v in b_actions.data.items()},
                return_aux=True,
            )

            bc_loss = torch.zeros((), device=device)
            entropy_loss = torch.zeros((), device=device)
            for agent_name in _ordered_agent_names(agent, b_actions):
                bc_loss = bc_loss + (-new_logp[agent_name].mean())
                entropy_loss = entropy_loss + entropy[agent_name].mean()

            future_state_loss = _future_state_aux_loss(args, aux_outputs, meta, mb_inds, batch_size, device)
            loss = bc_coef * bc_loss + future_state_coef * future_state_loss
            loss = loss / args.grad_accum_steps
            accelerator.backward(loss)
            accum_count += 1

            if accum_count % args.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(get_trainable_optimizer_parameters(optimizer), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                _writer_add_scalar(writer, "training_sft/bc_loss", bc_loss.item(), n_updates)
                _writer_add_scalar(writer, "training_sft/future_state_loss", future_state_loss.item(), n_updates)
                _writer_add_scalar(writer, "training_sft/entropy", entropy_loss.item(), n_updates)
                _writer_add_scalar(writer, "training_sft/loss", loss.item(), n_updates)
                stats["bc_loss"] += bc_loss.item()
                stats["future_state_loss"] += future_state_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["total_loss"] += loss.item()
                n_updates += 1
            pbar.update(1)

    for key in stats:
        stats[key] /= (n_updates + 1e-8)
    return stats


def maple_update_on_policy(args, agent, optimizer, data, collate_fn, accelerator, stage, writer, rollouts_num_aft_env_change=None):
    (
        b_obs,
        b_actions,
        b_logprobs,
        b_advantages,
        b_returns,
        b_values,
        meta,
    ) = _unpack_training_data(data)

    device = accelerator.device
    batch_size = b_advantages.shape[0]
    batch_indices = np.arange(batch_size)
    pbar = tqdm(total=args.update_epochs * max(1, batch_size // args.minibatch_size), desc=stage, ascii=True)

    shared_advantages = _reshape_meta_tensor(meta.get("shared_advantages"), batch_size, device, dtype=torch.float32)
    if shared_advantages is None:
        source = _reshape_meta_tensor(meta.get("joint_returns"), batch_size, device, dtype=torch.float32)
        if source is None:
            source = b_returns.to(device=device, dtype=torch.float32)
        group_ids = _reshape_meta_tensor(meta.get("group_ids"), batch_size, device, dtype=torch.long)
        if group_ids is None:
            group_size = int(getattr(args, "maple_group_size", getattr(args, "magrpo_group_size", 0)))
            if group_size > 0:
                group_ids = build_group_ids(batch_size, group_size, device)
        if group_ids is not None:
            shared_advantages = compute_group_relative_advantages(
                scores=source,
                group_ids=group_ids,
                eps=float(getattr(args, "maple_grpo_eps", getattr(args, "magrpo_eps", 1e-5))),
            )
        else:
            shared_advantages = (source - source.mean()) / (source.std(unbiased=False) + 1e-5)

    sample_mask = _reshape_meta_tensor(meta.get("sample_mask"), batch_size, device, dtype=torch.bool)
    sample_weights = _reshape_meta_tensor(meta.get("sample_weights"), batch_size, device, dtype=torch.float32)
    future_state_coef = float(getattr(args, "maple_future_state_coef", 0.0))
    bc_coef = float(getattr(args, "maple_rl_bc_coef", 0.0))
    entropy_coef = float(
        getattr(args, "maple_ent_coef", getattr(args, "magrpo_ent_coef", getattr(args, "ent_coef", 0.0)))
    )

    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "old_approx_kl": 0.0,
        "future_state_loss": 0.0,
        "bc_loss": 0.0,
        "adv_std": 0.0,
    }

    n_updates = 0
    accum_count = 0
    early_stop = False
    optimizer.zero_grad()

    for _ in range(args.update_epochs):
        np.random.shuffle(batch_indices)
        for start in range(0, batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = batch_indices[start:end]
            mb_inds_t = torch.as_tensor(mb_inds, device=device, dtype=torch.long)
            batch = collate_fn(b_obs[mb_inds])
            _, new_logp, entropy, new_value, aux_outputs = _policy_update_forward(
                agent,
                batch,
                {k: v[mb_inds] for k, v in b_actions.data.items()},
                return_aux=True,
            )
            new_value = new_value.view(-1)
            if (
                not torch.isfinite(new_value).all()
                or any(not torch.isfinite(v).all() for v in new_logp.values())
                or any(not torch.isfinite(v).all() for v in entropy.values())
            ):
                print(f"[MAPLE Skip] non-finite forward output at stage={stage}; skip minibatch")
                pbar.update(1)
                continue
            v_loss = _compute_value_loss(
                args,
                new_value,
                b_values[mb_inds].view(-1),
                b_returns[mb_inds].view(-1),
            )
            if not torch.isfinite(v_loss):
                print(f"[MAPLE Skip] non-finite value loss at stage={stage}; skip minibatch")
                pbar.update(1)
                continue

            mb_mask = sample_mask[mb_inds_t] if sample_mask is not None else None
            mb_weights = sample_weights[mb_inds_t] if sample_weights is not None else None
            mb_advantages = shared_advantages[mb_inds_t].to(device=device, dtype=torch.float32)

            if accum_count % args.grad_accum_steps == 0:
                run_old_kl = 0.0
                run_kl = 0.0
                run_clipfrac = 0.0
                run_count = 0

            policy_loss = torch.zeros((), device=device)
            entropy_loss = torch.zeros((), device=device)
            bc_loss = torch.zeros((), device=device)

            for agent_name in _ordered_agent_names(agent, b_actions):
                agent_mask = _meta_agent_vector_flat(meta, "agent_masks", agent_name, mb_inds, batch_size, device, dtype=torch.bool)
                if agent_mask is None:
                    agent_mask = mb_mask
                elif mb_mask is not None:
                    agent_mask = agent_mask & mb_mask

                agent_weights = _meta_agent_vector_flat(
                    meta,
                    "agent_weights",
                    agent_name,
                    mb_inds,
                    batch_size,
                    device,
                    dtype=torch.float32,
                )
                if agent_weights is None:
                    agent_weights = mb_weights

                logratio = new_logp[agent_name] - b_logprobs[agent_name][mb_inds].to(device)
                ratio = logratio.exp()
                pg_loss_1 = -mb_advantages * ratio
                pg_loss_2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                policy_loss = policy_loss + _masked_weighted_mean(torch.max(pg_loss_1, pg_loss_2), agent_mask, agent_weights)
                entropy_loss = entropy_loss + _masked_weighted_mean(entropy[agent_name], agent_mask, agent_weights)
                bc_loss = bc_loss + _masked_weighted_mean(-new_logp[agent_name], agent_mask, agent_weights)

                with torch.no_grad():
                    old_kl_mb = _masked_weighted_mean(-logratio, agent_mask, agent_weights).item()
                    new_kl_mb = _masked_weighted_mean((ratio - 1) - logratio, agent_mask, agent_weights).item()
                    clip_mb = _masked_weighted_mean(((ratio - 1).abs() > args.clip_eps).float(), agent_mask, agent_weights).item()
                    if agent_mask is not None:
                        micro_size = int(agent_mask.to(torch.int32).sum().item())
                    else:
                        micro_size = int(ratio.numel())
                    micro_size = max(micro_size, 1)
                    new_total = run_count + micro_size
                    run_old_kl += (old_kl_mb - run_old_kl) * (micro_size / new_total)
                    run_kl += (new_kl_mb - run_kl) * (micro_size / new_total)
                    run_clipfrac += (clip_mb - run_clipfrac) * (micro_size / new_total)
                    run_count = new_total

            future_state_loss = _future_state_aux_loss(args, aux_outputs, meta, mb_inds, batch_size, device)
            critic_only_update = getattr(args, "critic_only_update", False)
            if critic_only_update:
                loss = args.vf_coef * v_loss
            else:
                loss = (
                    policy_loss
                    - entropy_coef * entropy_loss
                    + args.vf_coef * v_loss
                    + bc_coef * bc_loss
                    + future_state_coef * future_state_loss
                )
            loss = loss / args.grad_accum_steps
            if not loss.requires_grad:
                raise RuntimeError(
                    "MAPLE loss has no grad at "
                    f"stage={stage}, critic_only_update={critic_only_update}, "
                    f"policy_requires_grad={policy_loss.requires_grad}, "
                    f"value_requires_grad={v_loss.requires_grad}, "
                    f"entropy_requires_grad={entropy_loss.requires_grad}, "
                    f"bc_requires_grad={bc_loss.requires_grad}, "
                    f"future_requires_grad={future_state_loss.requires_grad}, "
                    f"vf_coef={args.vf_coef}, bc_coef={bc_coef}, "
                    f"ent_coef={entropy_coef}, future_state_coef={future_state_coef}"
                )
            if not torch.isfinite(loss):
                print(f"[MAPLE Skip] non-finite loss at stage={stage}; skip minibatch")
                optimizer.zero_grad()
                pbar.update(1)
                continue
            accelerator.backward(loss)
            accum_count += 1

            if accum_count % args.grad_accum_steps == 0:
                approx_kl = run_kl
                old_approx_kl = run_old_kl
                clipfrac = run_clipfrac
                pbar.update(1)
                target_kl = getattr(args, "target_kl", None)
                if (
                    (not critic_only_update)
                    and rollouts_num_aft_env_change is not None
                    and rollouts_num_aft_env_change == -1
                    and target_kl is not None
                    and approx_kl > target_kl
                ):
                    print(
                        f"[MAPLE EarlyStop] stage={stage} "
                        f"approx_kl={approx_kl:.6f} target_kl={float(target_kl):.6f} "
                        f"old_approx_kl={old_approx_kl:.6f} clip_frac={clipfrac:.6f}"
                    )
                    early_stop = True
                    break

                nn.utils.clip_grad_norm_(get_trainable_optimizer_parameters(optimizer), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                centered_adv = mb_advantages if mb_mask is None else mb_advantages[mb_mask]
                adv_std = centered_adv.std(unbiased=False) if centered_adv.numel() > 0 else torch.zeros((), device=device)

                _writer_add_scalar(writer, "training/loss", loss.item(), n_updates)
                _writer_add_scalar(writer, "training/policy_loss", policy_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/value_loss", v_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/entropy", entropy_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/approx_kl", approx_kl, n_updates)
                _writer_add_scalar(writer, "training/old_approx_kl", old_approx_kl, n_updates)
                _writer_add_scalar(writer, "training/clip_frac", clipfrac, n_updates)
                _writer_add_scalar(writer, "training/future_state_loss", future_state_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/bc_loss", bc_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/adv_std", adv_std.item(), n_updates)

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += v_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clipfrac
                stats["old_approx_kl"] += old_approx_kl
                stats["future_state_loss"] += future_state_loss.item()
                stats["bc_loss"] += bc_loss.item()
                stats["adv_std"] += adv_std.item()
                n_updates += 1

        if early_stop:
            break

    for key in stats:
        stats[key] /= (n_updates + 1e-8)
    return stats
