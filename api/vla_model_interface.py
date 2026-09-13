"""Interface for integrating a multi-agent VLA model with ActExchanger."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.optim import Optimizer

from api.marl_method_interface import AgentModelInterface


@dataclass(frozen=True)
class VLAAgentSpec:
    """Agent and action-space configuration required by a VLA policy."""

    agent_names: tuple[str, ...]
    state_dims: Mapping[str, int]
    action_dims: Mapping[str, int]
    global_state_dim: int

    def __post_init__(self) -> None:
        if not self.agent_names:
            raise ValueError("agent_names must not be empty")
        if len(set(self.agent_names)) != len(self.agent_names):
            raise ValueError("agent_names must be unique")
        if self.global_state_dim <= 0:
            raise ValueError("global_state_dim must be positive")
        for name in self.agent_names:
            if self.state_dims.get(name, 0) <= 0:
                raise ValueError(f"state_dims[{name!r}] must be positive")
            if self.action_dims.get(name, 0) <= 0:
                raise ValueError(f"action_dims[{name!r}] must be positive")


class VLAModelInterface(AgentModelInterface, ABC):
    """Contract that exposes a VLA model to a multi-agent online-RL method.

    The implementation owns model-specific work: model construction, conversion of raw
    workload observations to a policy batch, action/value inference, optimization, and
    checkpoint serialization. MARL-specific planning, communication, rollout scheduling,
    and policy updates remain outside this interface.
    """

    @property
    @abstractmethod
    def agent_spec(self) -> VLAAgentSpec:
        """Return the agent and state/action configuration for this VLA model."""

    @property
    def agent_names(self) -> Sequence[str]:
        """Expose VLA agents through the common agent-model interface."""
        return self.agent_spec.agent_names

    @property
    @abstractmethod
    def policy_class(self) -> type[nn.Module]:
        """Return the concrete multi-agent VLA policy class."""

    @abstractmethod
    def configure_trainable_modules(
        self,
        policy: nn.Module,
        *,
        freeze_vla_backbone: bool,
    ) -> None:
        """Set trainability for the VLA backbone and task-specific modules."""

    def update_state_stats(self, policy: nn.Module, obs: Any) -> None:
        """Update optional policy observation normalization statistics."""
        update = getattr(policy, "update_state_stats", None)
        if update is not None:
            update(obs)

    def checkpoint_state_dict(self, policy: nn.Module) -> Mapping[str, torch.Tensor]:
        """Return the policy state to store in a checkpoint."""
        state_fn = getattr(policy, "checkpoint_state_dict", None)
        return state_fn() if state_fn is not None else policy.state_dict()

    def load_checkpoint_state_dict(
        self,
        policy: nn.Module,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Load a previously saved policy state."""
        load_fn = getattr(policy, "load_checkpoint_state_dict", None)
        if load_fn is not None:
            load_fn(dict(state_dict))
        else:
            policy.load_state_dict(dict(state_dict))

    def save_checkpoint(
        self,
        path: Path,
        policy: nn.Module,
        *,
        optimizer: Optional[Optimizer] = None,
        extra_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Save policy, optional optimizer, and method metadata."""
        payload: dict[str, Any] = {
            "model_name": self.model_name,
            "agent_names": self.agent_spec.agent_names,
            "policy": self.checkpoint_state_dict(policy),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if extra_state is not None:
            payload["extra_state"] = dict(extra_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
