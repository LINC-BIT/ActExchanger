from typing import Mapping, Sequence

import torch

from train.marl.magrpo.base import build_group_ids, compute_group_relative_advantages


def _compute_joint_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards, dtype=torch.float32)
    running = torch.zeros(rewards.shape[1], device=rewards.device, dtype=torch.float32)
    for step in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[step].to(torch.float32) + gamma * running * (1.0 - dones[step].to(torch.float32))
        returns[step] = running
    return returns


def _build_stepwise_group_ids(
    num_steps: int,
    num_envs: int,
    group_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if group_size <= 0:
        return None
    groups_per_step = max(1, (num_envs + group_size - 1) // group_size)
    local_group_ids = build_group_ids(num_envs, group_size, device=device)
    return torch.stack(
        [local_group_ids + step * groups_per_step for step in range(num_steps)],
        dim=0,
    )


def _normalize_scores(
    scores: torch.Tensor,
    sample_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    valid_scores = scores[sample_mask]
    mean = valid_scores.mean()
    std = valid_scores.std(unbiased=False)
    return (scores - mean) / (std + eps)


def _common_rollout_meta(
    agent_names: Sequence[str],
    rewards: torch.Tensor,
) -> dict[str, torch.Tensor | Mapping[str, torch.Tensor]]:
    sample_mask = torch.ones_like(rewards, dtype=torch.bool, device=rewards.device)
    agent_masks = {
        name: torch.ones_like(rewards, dtype=torch.bool, device=rewards.device)
        for name in agent_names
    }
    return {
        "sample_mask": sample_mask,
        "agent_masks": agent_masks,
    }


def build_magrpo_rollout_meta(
    args,
    *,
    agent_names: Sequence[str],
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> dict[str, torch.Tensor | Mapping[str, torch.Tensor] | None]:
    meta = _common_rollout_meta(agent_names, rewards)
    group_ids = _build_stepwise_group_ids(
        num_steps=rewards.shape[0],
        num_envs=rewards.shape[1],
        group_size=int(getattr(args, "magrpo_group_size", 0)),
        device=rewards.device,
    )
    meta["group_ids"] = group_ids
    return meta


def build_maporl_rollout_meta(
    args,
    *,
    agent_names: Sequence[str],
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> dict[str, torch.Tensor | Mapping[str, torch.Tensor]]:
    meta = _common_rollout_meta(agent_names, rewards)
    team_returns = _compute_joint_returns(
        rewards=rewards,
        dones=dones,
        gamma=float(getattr(args, "gamma", 0.8)),
    )
    meta["team_advantages"] = _normalize_scores(
        team_returns.to(torch.float32),
        meta["sample_mask"],
        eps=1e-5,
    )
    return meta


def build_mpdf_rollout_meta(
    args,
    *,
    agent_names: Sequence[str],
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> dict[str, torch.Tensor | Mapping[str, torch.Tensor] | None]:
    del dones
    meta = _common_rollout_meta(agent_names, rewards)
    group_ids = _build_stepwise_group_ids(
        num_steps=rewards.shape[0],
        num_envs=rewards.shape[1],
        group_size=int(getattr(args, "mpdf_group_size", 0)),
        device=rewards.device,
    )
    meta["group_ids"] = group_ids
    return meta
