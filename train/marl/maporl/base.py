import inspect
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from train.marl.mappo.base import collect_rollout, get_trainable_optimizer_parameters
from train.reinforcement_learning.utils import compute_gae


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


def _normalize_advantages(args, advantages: torch.Tensor) -> torch.Tensor:
    if getattr(args, "not_normalize_adv", False):
        return advantages
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def _target_kl(args, target_kl_override=None):
    if target_kl_override is not None:
        return target_kl_override
    return getattr(args, "target_kl", None)


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
        raise ValueError("MAPORL update expects at least 6 training tensors")

    meta = {}
    if len(data) >= 7 and isinstance(data[6], Mapping):
        meta = dict(data[6])

    return (*data[:6], meta)


def _meta_vector(meta: Mapping[str, Any], key: str, mb_inds, device, *, dtype=None):
    value = meta.get(key)
    if value is None:
        return None
    if isinstance(value, Mapping):
        raise TypeError(f"Expected vector-like meta field for '{key}', got mapping")
    tensor = _to_device_tensor(value, device, dtype=dtype)
    return tensor[mb_inds]


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
    return tensor[mb_inds]


def _combine_masks(base_mask: Optional[torch.Tensor], extra_mask: Optional[torch.Tensor]):
    if base_mask is None:
        return extra_mask
    if extra_mask is None:
        return base_mask
    return base_mask & extra_mask.to(device=base_mask.device, dtype=torch.bool)


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


def score_rule(
    rule_horizon: str,
    rule_agent_share: str,
    discount_factor: float,
    scores_turn_agent: Mapping[Tuple[int, int], torch.Tensor],
    total_rounds: int,
    total_agents: int,
    finished_question: Sequence[int],
    turn: int,
    agent: int,
):
    def get_score_for_turn(q: int, t: int, a: int):
        if finished_question[q] != -1 and t > finished_question[q]:
            return torch.tensor(-1.0)

        valid_indices = [idx for idx, f in enumerate(finished_question) if f == -1 or f >= t]
        q_index = valid_indices.index(q)
        return scores_turn_agent[(t, a)][q_index]

    if rule_horizon == "last" and rule_agent_share == "all":
        return torch.stack(
            [
                torch.stack([get_score_for_turn(q, total_rounds - 1, a) for a in range(total_agents)]).mean()
                for q in range(len(finished_question))
            ]
        )

    if rule_horizon == "last" and rule_agent_share == "individual":
        return torch.stack(
            [get_score_for_turn(q, total_rounds - 1, agent) for q in range(len(finished_question))]
        )

    if rule_horizon == "discounted_sum":
        outputs = []
        for q in range(len(finished_question)):
            last_turn = total_rounds - 1 if finished_question[q] == -1 else finished_question[q]
            if turn > last_turn:
                outputs.append(torch.tensor(-1.0))
                continue

            discount_terms = torch.tensor(
                [discount_factor ** (t_prime - turn) for t_prime in range(turn, last_turn + 1)],
                dtype=torch.float32,
            )
            discount_sum = discount_terms.sum().clamp_min(1e-8)

            if rule_agent_share == "all":
                current_score = get_score_for_turn(q, turn, agent)
                if turn != total_rounds - 1:
                    future_scores = []
                    for t_prime in range(turn + 1, last_turn + 1):
                        shared_score = torch.stack(
                            [
                                discount_factor ** (t_prime - turn) * get_score_for_turn(q, t_prime, a)
                                for a in range(total_agents)
                            ]
                        ).mean()
                        future_scores.append(shared_score)
                    future_score = torch.stack(future_scores).sum() if future_scores else torch.tensor(0.0)
                else:
                    future_score = torch.tensor(0.0)
                outputs.append((current_score + future_score) / discount_sum)
            elif rule_agent_share == "individual":
                score_sum = torch.stack(
                    [
                        get_score_for_turn(q, t_prime, agent) * discount_factor ** (t_prime - turn)
                        for t_prime in range(turn, last_turn + 1)
                    ]
                ).sum()
                outputs.append(score_sum / discount_sum)
            else:
                raise ValueError(f"Unsupported MAPORL rule_agent_share: {rule_agent_share}")

        return torch.stack(outputs)

    if rule_horizon == "current" and rule_agent_share == "individual":
        return torch.stack([get_score_for_turn(q, turn, agent) for q in range(len(finished_question))])

    if rule_horizon == "current" and rule_agent_share == "all":
        return torch.stack(
            [
                torch.stack([get_score_for_turn(q, turn, a) for a in range(total_agents)]).mean()
                for q in range(len(finished_question))
            ]
        )

    raise ValueError(f"Unsupported MAPORL horizon/share combination: {rule_horizon}/{rule_agent_share}")


def bonus_rule(
    correctnesses_turn_agent: Mapping[Tuple[int, int], torch.Tensor],
    total_rounds: int,
    total_agents: int,
    finished_question: Sequence[int],
    turn: int,
    agent: int,
    alpha: Sequence[float],
    correct_threshold: float = 0.5,
    wrong_threshold: float = 0.5,
):
    alpha = list(alpha)
    if len(alpha) < 4:
        alpha = alpha + [0.0] * (4 - len(alpha))

    def get_score_for_turn(q: int, t: int, a: int):
        if finished_question[q] != -1 and t > finished_question[q]:
            return torch.tensor(-1.0)

        valid_indices = [idx for idx, f in enumerate(finished_question) if f == -1 or f >= t]
        q_index = valid_indices.index(q)
        return correctnesses_turn_agent[(t, a)][q_index]

    base_scores = torch.stack([get_score_for_turn(q, turn, agent) for q in range(len(finished_question))])
    shaped = torch.stack(
        [torch.tensor(-1.0) if float(score.item()) == -1.0 else torch.tensor(0.0) for score in base_scores]
    )

    if turn == 0:
        return shaped

    for q in range(len(base_scores)):
        prev_score = float(get_score_for_turn(q, turn - 1, agent).item())
        current_score = float(get_score_for_turn(q, turn, agent).item())
        prev_others_score = None

        if total_agents > 1:
            prev_others_score = float(
                torch.stack(
                [get_score_for_turn(q, turn - 1, a) for a in range(total_agents) if a != agent]
                ).mean().item()
            )

        if prev_score > correct_threshold and current_score < wrong_threshold:
            if total_agents > 1 and prev_others_score is not None:
                if prev_others_score > correct_threshold:
                    shaped[q] -= alpha[1]
                elif prev_others_score < wrong_threshold:
                    shaped[q] -= alpha[0]
            else:
                shaped[q] -= alpha[1]

        if prev_score < wrong_threshold and current_score > correct_threshold:
            if total_agents > 1 and prev_others_score is not None:
                if prev_others_score < wrong_threshold:
                    shaped[q] += alpha[1]
                elif prev_others_score > correct_threshold:
                    shaped[q] += alpha[0]
            else:
                shaped[q] += alpha[1]

    if turn < total_rounds - 1 and total_agents > 1:
        for q in range(len(base_scores)):
            next_others_score = float(
                torch.stack(
                [get_score_for_turn(q, turn + 1, a) for a in range(total_agents) if a != agent]
                ).mean().item()
            )
            current_score = float(get_score_for_turn(q, turn, agent).item())
            current_others_score = float(
                torch.stack(
                [get_score_for_turn(q, turn, a) for a in range(total_agents) if a != agent]
                ).mean().item()
            )

            if next_others_score > correct_threshold and current_others_score < wrong_threshold:
                if current_score > correct_threshold:
                    shaped[q] -= alpha[3]
                elif current_score < wrong_threshold:
                    shaped[q] -= alpha[2]

            if next_others_score < wrong_threshold and current_others_score > correct_threshold:
                if current_score < wrong_threshold:
                    shaped[q] += alpha[3]
                elif current_score > correct_threshold:
                    shaped[q] += alpha[2]

    return shaped


def build_maporl_shaped_scores(
    scores_turn_agent: Mapping[Tuple[int, int], torch.Tensor],
    correctnesses_turn_agent: Mapping[Tuple[int, int], torch.Tensor],
    finished_question: Sequence[int],
    *,
    round_num: int,
    agent_num: int,
    rule_horizon: str = "discounted_sum",
    rule_agent_share: str = "all",
    rule_discount: float = 0.3,
    alpha: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
):
    shaped_scores = {}
    for turn in range(round_num):
        for agent in range(agent_num):
            score = score_rule(
                rule_horizon=rule_horizon,
                rule_agent_share=rule_agent_share,
                discount_factor=rule_discount,
                scores_turn_agent=scores_turn_agent,
                total_rounds=round_num,
                total_agents=agent_num,
                finished_question=finished_question,
                turn=turn,
                agent=agent,
            )
            bonus = bonus_rule(
                correctnesses_turn_agent=correctnesses_turn_agent,
                total_rounds=round_num,
                total_agents=agent_num,
                finished_question=finished_question,
                turn=turn,
                agent=agent,
                alpha=alpha,
            )
            combined = score + bonus
            shaped_scores[(turn, agent)] = combined[score != -1]
    return shaped_scores


def maporl_update_on_policy(
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

    b_advantages = _normalize_advantages(args, b_advantages)
    batch_size = b_advantages.shape[0]
    batch_indices = np.arange(batch_size)

    sample_mask = _to_device_tensor(meta.get("sample_mask"), device, dtype=torch.bool)
    sample_weights = _to_device_tensor(meta.get("sample_weights"), device, dtype=torch.float32)
    team_advantages = _to_device_tensor(meta.get("team_advantages"), device, dtype=torch.float32)
    team_adv_coef = float(getattr(args, "maporl_team_adv_coef", 0.0))
    ref_kl_coef = float(getattr(args, "maporl_kl_coef", getattr(args, "kl_coef", 0.0)))

    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "old_approx_kl": 0.0,
        "ref_kl": 0.0,
        "team_adv_bonus": 0.0,
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
            batch = collate_fn(b_obs[mb_inds])

            stored_actions = {k: v[mb_inds] for k, v in b_actions.data.items()}
            _, new_logp, entropy, new_value = _policy_update_forward(agent, batch, stored_actions)
            new_value = new_value.view(-1)

            mb_mask = sample_mask[mb_inds] if sample_mask is not None else None
            mb_weights = sample_weights[mb_inds] if sample_weights is not None else None
            mb_returns = b_returns[mb_inds].to(device)
            v_loss = 0.5 * _masked_weighted_mean((new_value - mb_returns) ** 2, mb_mask, mb_weights)

            ref_logprobs = _maybe_get_reference_logprobs(agent, batch, stored_actions)

            if accum_count % args.grad_accum_steps == 0:
                run_old_kl = 0.0
                run_kl = 0.0
                run_clipfrac = 0.0
                run_count = 0

            policy_loss = torch.zeros((), device=device)
            entropy_loss = torch.zeros((), device=device)
            ref_kl_loss = torch.zeros((), device=device)
            team_adv_bonus_value = 0.0

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

                advantage = b_advantages[mb_inds].to(device)
                if team_advantages is not None and team_adv_coef != 0.0:
                    team_bonus = team_advantages[mb_inds]
                    advantage = advantage + team_adv_coef * team_bonus
                    team_adv_bonus_value += _masked_weighted_mean(team_bonus, agent_mask, agent_weights).item()

                agent_bonus = _meta_agent_vector(
                    meta,
                    "agent_advantages",
                    agent_name,
                    mb_inds,
                    device,
                    dtype=torch.float32,
                )
                if agent_bonus is not None:
                    advantage = advantage + agent_bonus

                logratio = new_logp[agent_name] - b_logprobs[agent_name][mb_inds].to(device)
                ratio = logratio.exp()

                pg_loss_1 = -advantage * ratio
                pg_loss_2 = -advantage * torch.clamp(
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

            loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
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

                _writer_add_scalar(writer, "training/loss", loss.item(), n_updates)
                _writer_add_scalar(writer, "training/policy_loss", policy_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/value_loss", v_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/entropy", entropy_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/approx_kl", approx_kl, n_updates)
                _writer_add_scalar(writer, "training/old_approx_kl", old_approx_kl, n_updates)
                _writer_add_scalar(writer, "training/clip_frac", clipfrac, n_updates)
                _writer_add_scalar(writer, "training/ref_kl", ref_kl_loss.item(), n_updates)
                _writer_add_scalar(writer, "training/team_adv_bonus", team_adv_bonus_value, n_updates)

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += v_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clipfrac
                stats["old_approx_kl"] += old_approx_kl
                stats["ref_kl"] += ref_kl_loss.item()
                stats["team_adv_bonus"] += team_adv_bonus_value
                n_updates += 1

            pbar.update(1)

        if early_stop:
            break

    for key in stats:
        stats[key] /= (n_updates + 1e-8)

    return stats


def maporl_update_on_policy_ag(
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
    return maporl_update_on_policy(
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
