import inspect
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from train.marl.mappo.base import _compute_value_loss, collect_rollout, get_trainable_optimizer_parameters
from train.reinforcement_learning.utils import compute_gae


def _target_kl(args, target_kl_override=None):
    return target_kl_override if target_kl_override is not None else getattr(args, "target_kl", None)


def _joint_entropy(entropy_dict):
    entropies = torch.stack(list(entropy_dict.values()), dim=-1)
    return entropies.mean(dim=-1).mean()


def _joint_logprob(logprob_dict):
    logprobs = torch.stack(list(logprob_dict.values()), dim=-1)
    return logprobs.mean(dim=-1)


def _agent_supports_kwarg(agent, kwarg_name):
    try:
        signature = inspect.signature(agent.get_action_and_value)
    except (TypeError, ValueError, AttributeError):
        return False
    return kwarg_name in signature.parameters


def mat_update_on_policy(
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
    ) = data

    batch_size = b_advantages.shape[0]
    batch_indices = np.arange(batch_size)

    pbar = tqdm(total=args.update_epochs * (batch_size // args.minibatch_size), desc=stage, ascii=True)
    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "old_approx_kl": 0.0,
        "full_kl": 0.0,
        "argmax_change_frac": 0.0,
    }

    n_updates = 0
    accum_count = 0
    early_stop = False
    critic_only_update = getattr(args, "critic_only_update", False)

    if not args.not_normalize_adv:
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

    optimizer.zero_grad()

    for _ in range(args.update_epochs):
        np.random.shuffle(batch_indices)

        for start in range(0, batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = batch_indices[start:end]
            batch = collate_fn(b_obs[mb_inds])

            if critic_only_update:
                if hasattr(agent, "get_trainable_value"):
                    new_value = agent.get_trainable_value(batch)
                else:
                    new_value = agent.get_value(batch)
                new_value = new_value.view(-1)
                if not torch.isfinite(new_value).all():
                    optimizer.zero_grad()
                    print(f"[{stage}] skip non-finite value forward output")
                    pbar.update(1)
                    continue
                new_logp = None
                entropy = None
            else:
                if _agent_supports_kwarg(agent, "action_bins_input"):
                    _, new_logp, entropy, new_value = agent.get_action_and_value(
                        batch,
                        action_bins_input={k: v[mb_inds] for k, v in b_actions.data.items()},
                    )
                else:
                    _, new_logp, entropy, new_value = agent.get_action_and_value(
                        batch,
                        {k: v[mb_inds] for k, v in b_actions.data.items()},
                    )

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

            v_loss = _compute_value_loss(
                args,
                new_value,
                b_values[mb_inds].view(-1),
                b_returns[mb_inds].view(-1),
            )
            if not torch.isfinite(v_loss):
                optimizer.zero_grad()
                print(f"[{stage}] skip non-finite value loss")
                pbar.update(1)
                continue

            if accum_count % args.grad_accum_steps == 0:
                run_old_kl = 0.0
                run_kl = 0.0
                run_clipfrac = 0.0
                run_count = 0

            if critic_only_update:
                policy_loss = torch.zeros((), device=new_value.device, dtype=new_value.dtype)
                entropy_loss = torch.zeros((), device=new_value.device, dtype=new_value.dtype)
            else:
                new_joint_logp = _joint_logprob(new_logp)
                old_joint_logp = _joint_logprob({k: v[mb_inds] for k, v in b_logprobs.data.items()})
                logratio = new_joint_logp - old_joint_logp
                ratio = logratio.exp()

                pg1 = -b_advantages[mb_inds] * ratio
                pg2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                policy_loss = torch.max(pg1, pg2).mean()
                entropy_loss = _joint_entropy(entropy)

                with torch.no_grad():
                    old_kl_mb = (-logratio).mean().item()
                    new_kl_mb = ((ratio - 1) - logratio).mean().item()
                    clip_mb = ((ratio - 1).abs() > args.clip_eps).float().mean().item()
                    micro_size = ratio.numel()
                    new_total = run_count + micro_size

                    run_old_kl += (old_kl_mb - run_old_kl) * (micro_size / new_total)
                    run_kl += (new_kl_mb - run_kl) * (micro_size / new_total)
                    run_clipfrac += (clip_mb - run_clipfrac) * (micro_size / new_total)
                    run_count = new_total

            if critic_only_update:
                loss = args.vf_coef * v_loss
            else:
                loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                print(
                    f"[{stage}] skip non-finite loss: "
                    f"loss={loss.detach().item() if loss.numel() == 1 else loss}, "
                    f"policy={policy_loss.detach().item() if torch.is_tensor(policy_loss) else policy_loss}, "
                    f"value={v_loss.detach().item() if torch.is_tensor(v_loss) else v_loss}, "
                    f"entropy={entropy_loss.detach().item() if torch.is_tensor(entropy_loss) else entropy_loss}"
                )
                pbar.update(1)
                continue
            loss = loss / args.grad_accum_steps

            accelerator.backward(loss)
            accum_count += 1

            pbar.update(1)

            if accum_count % args.grad_accum_steps == 0:
                approx_kl = run_kl
                old_approx_kl = run_old_kl
                clipfrac = run_clipfrac

                grad_norm = nn.utils.clip_grad_norm_(
                    get_trainable_optimizer_parameters(optimizer),
                    args.max_grad_norm,
                )
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad()
                    print(f"[{stage}] skip non-finite grad_norm={grad_norm}")
                    continue
                optimizer.step()
                optimizer.zero_grad()

                if writer is not None:
                    writer.add_scalar("training/loss", loss.item(), n_updates)
                    writer.add_scalar("training/policy_loss", policy_loss.item(), n_updates)
                    writer.add_scalar("training/value_loss", v_loss.item(), n_updates)
                    writer.add_scalar("training/entropy", entropy_loss.item(), n_updates)
                    writer.add_scalar("training/approx_kl", approx_kl, n_updates)
                    writer.add_scalar("training/old_approx_kl", old_approx_kl, n_updates)
                    writer.add_scalar("training/clip_frac", clipfrac, n_updates)

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += v_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clipfrac
                stats["old_approx_kl"] += old_approx_kl
                n_updates += 1

                if (
                    not critic_only_update
                    and
                    rollouts_num_aft_env_change is not None
                    and rollouts_num_aft_env_change == -1
                    and _target_kl(args, target_kl_override) is not None
                    and approx_kl > _target_kl(args, target_kl_override)
                ):
                    print(
                        f"[{stage}] early stop on KL: "
                        f"approx_kl={approx_kl:.6f}, "
                        f"old_approx_kl={old_approx_kl:.6f}, "
                        f"target={_target_kl(args, target_kl_override):.6f}, "
                        f"clip_frac={clipfrac:.6f}"
                    )
                    early_stop = True
                    break

        if early_stop:
            break

    for key in stats:
        stats[key] /= (n_updates + 1e-8)

    return stats


def mat_update_on_policy_ag(
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
    return mat_update_on_policy(
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
