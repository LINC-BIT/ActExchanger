import inspect
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from train.marl.mappo.base import collect_rollout, get_trainable_optimizer_parameters
from train.reinforcement_learning.utils import compute_gae


META_ACTIONS = ("persist", "refine", "concede")


def _agent_supports_kwarg(agent, kwarg_name: str) -> bool:
    try:
        signature = inspect.signature(agent.get_action_and_value)
    except (TypeError, ValueError, AttributeError):
        return False
    return kwarg_name in signature.parameters


def _ordered_agent_names(agent, b_actions=None, logprob_dict=None):
    if hasattr(agent, "actor_heads"):
        return list(agent.actor_heads.keys())
    if hasattr(agent, "agent_names"):
        return list(agent.agent_names)
    if b_actions is not None and hasattr(b_actions, "data"):
        return list(b_actions.data.keys())
    if logprob_dict is not None:
        return list(logprob_dict.keys())
    return []


def _policy_update_forward(agent, batch, stored_actions):
    if _agent_supports_kwarg(agent, "action_bins_input"):
        return agent.get_action_and_value(
            batch,
            action_bins_input={k: v for k, v in stored_actions.items()},
        )
    return agent.get_action_and_value(
        batch,
        actions_input={k: v for k, v in stored_actions.items()},
    )


def _writer_add_scalar(writer, name: str, value: float, step: int) -> None:
    if writer is None:
        return
    writer.add_scalar(name, value, step)


def _to_device_tensor(
    value: Optional[Any],
    device: torch.device,
    *,
    dtype: Optional[torch.dtype] = None,
):
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


def _target_kl(args, target_kl_override=None):
    if target_kl_override is not None:
        return target_kl_override
    return getattr(args, "target_kl", None)


def _maybe_get_reference_logprobs(agent, batch, stored_actions):
    if hasattr(agent, "get_reference_action_logprobs"):
        ref_fn = agent.get_reference_action_logprobs
        if _agent_supports_kwarg(agent, "action_bins_input"):
            return ref_fn(
                batch,
                action_bins_input={k: v for k, v in stored_actions.items()},
            )
        return ref_fn(
            batch,
            actions_input={k: v for k, v in stored_actions.items()},
        )
    return None


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

    if len(data) < 6:
        raise ValueError("MPDF update expects at least 6 training tensors")

    meta = {}
    if len(data) >= 7 and isinstance(data[6], Mapping):
        meta = dict(data[6])

    return (*data[:6], meta)


def _meta_agent_vector(meta: Mapping[str, Any], key: str, agent_name: str, mb_inds, device, *, dtype=None):
    container = meta.get(key)
    if container is None:
        return None
    if not isinstance(container, Mapping):
        raise TypeError(f"Expected mapping meta field for '{key}', got {type(container).__name__}")
    value = container.get(agent_name)
    if value is None:
        return None
    tensor = _to_device_tensor(value, device, dtype=dtype)
    if isinstance(mb_inds, torch.Tensor):
        index = mb_inds.to(device=device, dtype=torch.long)
    else:
        index = torch.as_tensor(mb_inds, device=device, dtype=torch.long)
    return tensor[index]


def _combine_masks(base_mask: Optional[torch.Tensor], extra_mask: Optional[torch.Tensor]):
    if base_mask is None:
        return extra_mask
    if extra_mask is None:
        return base_mask
    return base_mask & extra_mask.to(device=base_mask.device, dtype=torch.bool)


def build_group_ids(batch_size: int, group_size: int, device: torch.device) -> torch.Tensor:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return torch.arange(batch_size, device=device, dtype=torch.long) // group_size


def meta_action_to_id(action: str) -> int:
    normalized = action.strip().lower()
    if normalized not in META_ACTIONS:
        raise ValueError(f"Unsupported MPDF meta action: {action}")
    return META_ACTIONS.index(normalized)


def ids_to_meta_actions(action_ids: torch.Tensor) -> Sequence[str]:
    return [META_ACTIONS[int(idx)] for idx in action_ids.view(-1).tolist()]


def compute_soft_ranks(
    scores: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    scores = scores.to(dtype=torch.float32)
    tau = max(float(temperature), 1e-6)
    diffs = (scores.unsqueeze(0) - scores.unsqueeze(1)) / tau
    wins = torch.sigmoid(diffs)
    return wins.sum(dim=-1) + 0.5


def soft_rank_to_advantages(
    soft_ranks: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    soft_ranks = soft_ranks.to(dtype=torch.float32)
    count = max(int(soft_ranks.numel()), 1)
    percentiles = (soft_ranks - 0.5) / max(count, 1)
    percentiles = percentiles.clamp(eps, 1.0 - eps)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=soft_ranks.device),
        torch.tensor(1.0, device=soft_ranks.device),
    )
    return normal.icdf(percentiles)


def compute_softrank_advantages(
    scores: torch.Tensor,
    group_ids: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    scores = scores.to(dtype=torch.float32)
    if group_ids is None:
        group_ids = torch.zeros(scores.shape[0], device=scores.device, dtype=torch.long)
    else:
        group_ids = group_ids.to(device=scores.device, dtype=torch.long)

    if mask is None:
        valid_mask = torch.ones_like(group_ids, dtype=torch.bool, device=scores.device)
    else:
        valid_mask = mask.to(device=scores.device, dtype=torch.bool)

    advantages = torch.zeros_like(scores, dtype=torch.float32)
    for group_id in torch.unique(group_ids[valid_mask]):
        group_mask = valid_mask & (group_ids == group_id)
        group_scores = scores[group_mask]
        if group_scores.numel() == 0:
            continue
        ranks = compute_soft_ranks(group_scores, temperature=temperature)
        advantages[group_mask] = soft_rank_to_advantages(ranks, eps=eps)
    return advantages


def build_softrank_targets(
    rewards: torch.Tensor,
    group_ids: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    eps: float = 1e-5,
):
    rewards = rewards.to(dtype=torch.float32)
    advantages = compute_softrank_advantages(
        scores=rewards,
        group_ids=group_ids,
        mask=mask,
        temperature=temperature,
        eps=eps,
    )
    return advantages, rewards


def _resolve_softrank_advantages(
    args,
    b_advantages: torch.Tensor,
    b_returns: torch.Tensor,
    meta: Mapping[str, Any],
    device: torch.device,
):
    eps = float(getattr(args, "mpdf_eps", 1e-5))
    temperature = float(getattr(args, "mpdf_rank_temperature", 1.0))

    sample_mask = _to_device_tensor(meta.get("sample_mask"), device, dtype=torch.bool)
    group_ids = _to_device_tensor(meta.get("group_ids"), device, dtype=torch.long)
    rewards = _to_device_tensor(meta.get("rewards"), device, dtype=torch.float32)
    shared_advantages = _to_device_tensor(meta.get("shared_advantages"), device, dtype=torch.float32)

    if group_ids is None:
        group_size = int(getattr(args, "mpdf_group_size", 0))
        if group_size > 0:
            group_ids = build_group_ids(b_advantages.shape[0], group_size, device)

    if shared_advantages is not None:
        return shared_advantages, rewards, group_ids, sample_mask

    if rewards is None:
        if b_returns is not None:
            rewards = b_returns.to(device=device, dtype=torch.float32)
        else:
            rewards = b_advantages.to(device=device, dtype=torch.float32)

    advantages = compute_softrank_advantages(
        scores=rewards,
        group_ids=group_ids,
        mask=sample_mask,
        temperature=temperature,
        eps=eps,
    )
    return advantages, rewards, group_ids, sample_mask


def mpdf_update_on_policy(
    args,
    agent,
    optimizer,
    data,
    collate_fn,
    accelerator,
    stage,
    writer,
    rollouts_num_aft_env_change=None,
    target_kl_override=None,
):
    (
        b_obs,
        b_actions,
        b_logprobs,
        b_advantages,
        b_returns,
        b_values,
        meta,
    ) = _unpack_training_data(data)

    del b_values
    device = accelerator.device
    batch_size = b_advantages.shape[0]
    batch_indices = np.arange(batch_size)

    shared_advantages, rewards, group_ids, sample_mask = _resolve_softrank_advantages(
        args,
        b_advantages=b_advantages,
        b_returns=b_returns,
        meta=meta,
        device=device,
    )
    sample_weights = _to_device_tensor(meta.get("sample_weights"), device, dtype=torch.float32)
    rank_temperature = float(getattr(args, "mpdf_rank_temperature", 1.0))
    ref_kl_coef = float(getattr(args, "mpdf_kl_coef", getattr(args, "kl_coef", 0.0)))
    entropy_coef = float(getattr(args, "mpdf_ent_coef", getattr(args, "ent_coef", 0.0)))

    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "old_approx_kl": 0.0,
        "ref_kl": 0.0,
        "reward_mean": 0.0,
        "adv_std": 0.0,
    }

    pbar = tqdm(
        total=args.update_epochs * (batch_size // args.minibatch_size),
        desc=stage,
        ascii=True,
    )

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

            stored_actions = {k: v[mb_inds] for k, v in b_actions.data.items()}
            _, new_logp, entropy, _ = _policy_update_forward(agent, batch, stored_actions)
            ref_logprobs = _maybe_get_reference_logprobs(agent, batch, stored_actions)

            mb_mask = sample_mask[mb_inds_t] if sample_mask is not None else None
            mb_weights = sample_weights[mb_inds_t] if sample_weights is not None else None
            mb_advantages = shared_advantages[mb_inds_t].to(device=device, dtype=torch.float32)
            mb_rewards = rewards[mb_inds_t].to(device=device, dtype=torch.float32)

            if accum_count % args.grad_accum_steps == 0:
                run_old_kl = 0.0
                run_kl = 0.0
                run_clipfrac = 0.0
                run_count = 0

            policy_loss = torch.zeros((), device=device)
            entropy_loss = torch.zeros((), device=device)
            ref_kl_loss = torch.zeros((), device=device)

            for agent_name in _ordered_agent_names(agent, b_actions=b_actions, logprob_dict=new_logp):
                agent_mask = _meta_agent_vector(meta, "agent_masks", agent_name, mb_inds, device, dtype=torch.bool)
                agent_mask = _combine_masks(mb_mask, agent_mask)

                agent_weights = _meta_agent_vector(
                    meta,
                    "agent_weights",
                    agent_name,
                    mb_inds,
                    device,
                    dtype=torch.float32,
                )
                if agent_weights is None:
                    agent_weights = mb_weights

                logratio = new_logp[agent_name] - b_logprobs[agent_name][mb_inds].to(device)
                ratio = logratio.exp()

                pg_loss_1 = -mb_advantages * ratio
                pg_loss_2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - args.clip_eps,
                    1 + args.clip_eps,
                )
                policy_loss = policy_loss + _masked_weighted_mean(
                    torch.max(pg_loss_1, pg_loss_2),
                    agent_mask,
                    agent_weights,
                )
                entropy_loss = entropy_loss + _masked_weighted_mean(
                    entropy[agent_name],
                    agent_mask,
                    agent_weights,
                )

                if ref_logprobs is not None and agent_name in ref_logprobs:
                    ref_kl = new_logp[agent_name] - ref_logprobs[agent_name].to(device)
                    ref_kl_loss = ref_kl_loss + _masked_weighted_mean(ref_kl, agent_mask, agent_weights)

                with torch.no_grad():
                    old_kl_mb = _masked_weighted_mean(-logratio, agent_mask, agent_weights).item()
                    new_kl_mb = _masked_weighted_mean((ratio - 1) - logratio, agent_mask, agent_weights).item()
                    clip_mb = _masked_weighted_mean(
                        ((ratio - 1).abs() > args.clip_eps).float(),
                        agent_mask,
                        agent_weights,
                    ).item()

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

            loss = policy_loss - entropy_coef * entropy_loss
            if ref_logprobs is not None and ref_kl_coef != 0.0:
                loss = loss + ref_kl_coef * ref_kl_loss

            loss = loss / args.grad_accum_steps
            accelerator.backward(loss)
            accum_count += 1

            if accum_count % args.grad_accum_steps == 0:
                approx_kl = run_kl
                old_approx_kl = run_old_kl
                clipfrac = run_clipfrac

                if (
                    rollouts_num_aft_env_change is not None
                    and rollouts_num_aft_env_change == -1
                    and _target_kl(args, target_kl_override) is not None
                    and approx_kl > _target_kl(args, target_kl_override)
                ):
                    early_stop = True
                    break

                nn.utils.clip_grad_norm_(
                    get_trainable_optimizer_parameters(optimizer),
                    args.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad()

                reward_mean = _masked_weighted_mean(mb_rewards, mb_mask, mb_weights)
                adv_mean = _masked_weighted_mean(mb_advantages, mb_mask, mb_weights)
                adv_std = _masked_weighted_mean((mb_advantages - adv_mean) ** 2, mb_mask, mb_weights).sqrt()

                _writer_add_scalar(writer, "training/loss", loss.item(), n_updates)
                _writer_add_scalar(writer, "training/policy_loss", policy_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/value_loss", 0.0, n_updates)
                _writer_add_scalar(writer, "training/entropy", entropy_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/approx_kl", approx_kl, n_updates)
                _writer_add_scalar(writer, "training/old_approx_kl", old_approx_kl, n_updates)
                _writer_add_scalar(writer, "training/clip_frac", clipfrac, n_updates)
                _writer_add_scalar(writer, "training/ref_kl", ref_kl_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/reward_mean", reward_mean.item(), n_updates)
                _writer_add_scalar(writer, "training/adv_std", adv_std.item(), n_updates)
                _writer_add_scalar(writer, "training/softrank_temperature", rank_temperature, n_updates)

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += 0.0
                stats["entropy"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clipfrac
                stats["old_approx_kl"] += old_approx_kl
                stats["ref_kl"] += ref_kl_loss.item()
                stats["reward_mean"] += reward_mean.item()
                stats["adv_std"] += adv_std.item()
                n_updates += 1

            pbar.update(1)

        if early_stop:
            break

    for key in stats:
        stats[key] /= (n_updates + 1e-8)

    return stats


def mpdf_update_on_policy_ag(
    args,
    agent,
    optimizer,
    data,
    collate_fn,
    accelerator,
    stage,
    clients,
    writer,
    rollouts_num_aft_env_change=None,
    target_kl_override=None,
):
    del clients
    return mpdf_update_on_policy(
        args,
        agent,
        optimizer,
        data,
        collate_fn,
        accelerator,
        stage,
        writer,
        rollouts_num_aft_env_change=rollouts_num_aft_env_change,
        target_kl_override=target_kl_override,
    )
