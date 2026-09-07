import copy
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from train.marl.magrpo.base import (
    _combine_masks,
    _masked_weighted_mean,
    _maybe_get_reference_logprobs,
    _meta_agent_vector,
    _ordered_agent_names,
    _policy_update_forward,
    _resolve_shared_advantages,
    _target_kl,
    _to_device_tensor,
    _unpack_training_data,
    _writer_add_scalar,
    build_group_ids,
    build_magrpo_advantages,
    compute_discounted_group_returns,
    compute_group_relative_advantages,
)
from train.marl.mappo.base import _compute_value_loss, collect_rollout, get_trainable_optimizer_parameters
from train.reinforcement_learning.utils import compute_gae


def _stage_key(stage: Optional[str]) -> str:
    normalized = (stage or "").strip().lower()
    if "critic" in normalized:
        return "critic"
    if "ppo stage 1" in normalized:
        return "stage1"
    if "ppo stage 2" in normalized:
        return "stage2"
    return "default"


def _stage_value(args, name: str, stage: Optional[str], default: float) -> float:
    key = _stage_key(stage)
    if key != "default":
        for attr in (
            f"comatrack_{key}_{name}",
            f"comatrack_{name}_{key}",
        ):
            if hasattr(args, attr):
                return float(getattr(args, attr))

    for attr in (
        f"comatrack_{name}",
        f"magrpo_{name}",
        name,
    ):
        if hasattr(args, attr):
            return float(getattr(args, attr))
    return float(default)


def _comatrack_args_view(args, stage: Optional[str]):
    view = copy.copy(args)
    view.magrpo_eps = getattr(args, "comatrack_eps", getattr(args, "magrpo_eps", 1e-5))
    view.magrpo_gamma = getattr(
        args,
        "comatrack_gamma",
        getattr(args, "magrpo_gamma", getattr(args, "gamma", 1.0)),
    )
    view.magrpo_group_size = getattr(
        args,
        "comatrack_group_size",
        getattr(args, "magrpo_group_size", 0),
    )
    view.magrpo_kl_coef = _stage_value(args, "kl_coef", stage, 0.0)
    view.magrpo_ent_coef = _stage_value(args, "ent_coef", stage, getattr(args, "ent_coef", 0.0))
    view.target_kl = getattr(
        args,
        "comatrack_target_kl",
        getattr(args, "magrpo_target_kl", getattr(args, "target_kl", None)),
    )
    return view


def build_comatrack_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    turn_ids: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    return build_magrpo_advantages(
        rewards=rewards,
        group_ids=group_ids,
        turn_ids=turn_ids,
        gamma=gamma,
        mask=mask,
        eps=eps,
    )


def comatrack_update_on_policy(
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

    device = accelerator.device
    batch_size = b_advantages.shape[0]
    batch_indices = np.arange(batch_size)

    view_args = _comatrack_args_view(args, stage)
    has_comatrack_meta = any(
        key in meta
        for key in (
            "shared_advantages",
            "joint_returns",
            "joint_rewards",
            "group_ids",
            "turn_ids",
            "sample_mask",
        )
    ) or int(getattr(view_args, "magrpo_group_size", 0)) > 0
    if has_comatrack_meta:
        shared_advantages, _, sample_mask = _resolve_shared_advantages(
            view_args,
            b_advantages=b_advantages,
            b_returns=b_returns,
            meta=meta,
            device=device,
        )
    else:
        # Plain MAPPO rollouts do not carry CoMaTrack grouping metadata. In that
        # case, using batch-centered returns makes good-but-below-average
        # trajectories negative and can quickly destroy a warm-started policy.
        shared_advantages = b_advantages.to(device=device, dtype=torch.float32)
        sample_mask = None
    if not getattr(args, "not_normalize_adv", False):
        adv_for_stats = shared_advantages
        if sample_mask is not None:
            adv_for_stats = adv_for_stats[sample_mask.to(device=device, dtype=torch.bool)]
        if adv_for_stats.numel() > 1:
            shared_advantages = (shared_advantages - adv_for_stats.mean()) / (adv_for_stats.std() + 1e-8)
    sample_weights = _to_device_tensor(meta.get("sample_weights"), device, dtype=torch.float32)

    ref_kl_coef = float(view_args.magrpo_kl_coef)
    entropy_coef = float(view_args.magrpo_ent_coef)
    value_coef = _stage_value(args, "value_coef", stage, getattr(args, "vf_coef", 0.5))

    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "old_approx_kl": 0.0,
        "ref_kl": 0.0,
        "adv_std": 0.0,
        "full_kl": 0.0,
        "argmax_change_frac": 0.0,
    }

    pbar = tqdm(
        total=args.update_epochs * max(batch_size // args.minibatch_size, 1),
        desc=stage,
        ascii=True,
    )

    n_updates = 0
    accum_count = 0
    early_stop = False
    critic_only_update = getattr(args, "critic_only_update", False) or _stage_key(stage) == "critic"
    optimizer.zero_grad()

    for _ in range(args.update_epochs):
        np.random.shuffle(batch_indices)

        for start in range(0, batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = batch_indices[start:end]
            mb_inds_t = torch.as_tensor(mb_inds, device=device, dtype=torch.long)
            batch = collate_fn(b_obs[mb_inds])

            stored_actions = {k: v[mb_inds] for k, v in b_actions.data.items()}
            _, new_logp, entropy, new_value = _policy_update_forward(agent, batch, stored_actions)
            ref_logprobs = _maybe_get_reference_logprobs(agent, batch, stored_actions)
            new_value = new_value.view(-1)

            if (
                not torch.isfinite(new_value).all()
                or any(not torch.isfinite(v).all() for v in new_logp.values())
                or any(not torch.isfinite(v).all() for v in entropy.values())
            ):
                optimizer.zero_grad()
                print(f"[{stage}] skip non-finite forward output")
                pbar.update(1)
                continue

            mb_mask = sample_mask[mb_inds_t] if sample_mask is not None else None
            mb_weights = sample_weights[mb_inds_t] if sample_weights is not None else None
            mb_advantages = shared_advantages[mb_inds_t].to(device=device, dtype=torch.float32)
            mb_returns = b_returns[mb_inds].to(device=device, dtype=torch.float32)

            if accum_count % args.grad_accum_steps == 0:
                run_old_kl = 0.0
                run_kl = 0.0
                run_clipfrac = 0.0
                run_count = 0

            policy_loss = torch.zeros((), device=device)
            entropy_loss = torch.zeros((), device=device)
            ref_kl_loss = torch.zeros((), device=device)
            active_agent_count = 0

            for agent_name in _ordered_agent_names(agent, b_actions=b_actions, logprob_dict=new_logp):
                active_agent_count += 1
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

            active_agent_count = max(active_agent_count, 1)
            policy_loss = policy_loss / active_agent_count
            entropy_loss = entropy_loss / active_agent_count
            ref_kl_loss = ref_kl_loss / active_agent_count

            value_loss = torch.zeros((), device=device)
            if value_coef != 0.0:
                value_loss = _compute_value_loss(
                    args,
                    new_value,
                    b_values[mb_inds].view(-1),
                    mb_returns,
                )
                if not torch.isfinite(value_loss):
                    optimizer.zero_grad()
                    print(f"[{stage}] skip non-finite value loss")
                    pbar.update(1)
                    continue

            if critic_only_update:
                loss = value_coef * value_loss
            else:
                loss = policy_loss - entropy_coef * entropy_loss + value_coef * value_loss
            if not critic_only_update and ref_logprobs is not None and ref_kl_coef != 0.0:
                loss = loss + ref_kl_coef * ref_kl_loss

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                print(
                    f"[{stage}] skip non-finite loss: "
                    f"loss={loss.detach().item() if loss.numel() == 1 else loss}, "
                    f"policy={policy_loss.detach().item() if torch.is_tensor(policy_loss) else policy_loss}, "
                    f"value={value_loss.detach().item() if torch.is_tensor(value_loss) else value_loss}, "
                    f"entropy={entropy_loss.detach().item() if torch.is_tensor(entropy_loss) else entropy_loss}"
                )
                pbar.update(1)
                continue

            loss = loss / args.grad_accum_steps
            accelerator.backward(loss)
            accum_count += 1

            if accum_count % args.grad_accum_steps == 0:
                approx_kl = run_kl
                old_approx_kl = run_old_kl
                clipfrac = run_clipfrac

                if (
                    not critic_only_update
                    and rollouts_num_aft_env_change is not None
                    and rollouts_num_aft_env_change == -1
                    and _target_kl(view_args, target_kl_override) is not None
                    and approx_kl > _target_kl(view_args, target_kl_override)
                ):
                    print(
                        f"[{stage}] early stop on KL: "
                        f"approx_kl={approx_kl:.6f}, "
                        f"old_approx_kl={old_approx_kl:.6f}, "
                        f"target={_target_kl(view_args, target_kl_override):.6f}, "
                        f"clip_frac={clipfrac:.6f}"
                    )
                    early_stop = True
                    break

                grad_norm = nn.utils.clip_grad_norm_(
                    get_trainable_optimizer_parameters(optimizer),
                    args.max_grad_norm,
                )
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad()
                    print(f"[{stage}] skip non-finite grad_norm={grad_norm}")
                    pbar.update(1)
                    continue
                optimizer.step()
                optimizer.zero_grad()

                centered_adv = mb_advantages
                if mb_mask is not None:
                    centered_adv = centered_adv[mb_mask]
                if centered_adv.numel() == 0:
                    mb_adv_std = torch.zeros((), device=device)
                else:
                    mb_adv_std = centered_adv.std(unbiased=False)

                _writer_add_scalar(writer, "training/loss", loss.item(), n_updates)
                _writer_add_scalar(writer, "training/policy_loss", policy_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/value_loss", value_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/entropy", entropy_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/approx_kl", approx_kl, n_updates)
                _writer_add_scalar(writer, "training/old_approx_kl", old_approx_kl, n_updates)
                _writer_add_scalar(writer, "training/clip_frac", clipfrac, n_updates)
                _writer_add_scalar(writer, "training/ref_kl", ref_kl_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/adv_std", mb_adv_std.item(), n_updates)

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clipfrac
                stats["old_approx_kl"] += old_approx_kl
                stats["ref_kl"] += ref_kl_loss.item()
                stats["adv_std"] += mb_adv_std.item()
                n_updates += 1

            pbar.update(1)

        if early_stop:
            break

    for key in stats:
        stats[key] /= (n_updates + 1e-8)

    return stats


def comatrack_update_on_policy_ag(
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
    return comatrack_update_on_policy(
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
