"""Reusable interfaces for integrating MARL methods with ActExchanger."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.optim import Optimizer


class AgentModelInterface(ABC):
    """Contract for one multi-agent method's agent-side policy model."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return a stable name used for checkpoints and experiment outputs."""

    @property
    @abstractmethod
    def agent_names(self) -> Sequence[str]:
        """Return the ordered names of agents controlled by the policy."""

    @abstractmethod
    def build_model(
        self,
        *,
        device: torch.device,
        config: Mapping[str, Any],
    ) -> nn.Module:
        """Construct the actor-critic model for the selected workload."""

    @abstractmethod
    def build_batch_from_obs(
        self,
        obs: Any,
        *,
        device: torch.device,
    ) -> Mapping[str, Any]:
        """Convert raw environment observations to the policy batch schema."""

    @abstractmethod
    def get_action_and_value(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        *,
        actions_input: Optional[Mapping[str, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[Mapping[str, Any], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], torch.Tensor]:
        """Return per-agent actions, log probabilities, entropies, and values."""

    @abstractmethod
    def get_action(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        *,
        deterministic: bool = False,
    ) -> Mapping[str, Any]:
        """Sample or select one action for every active agent."""

    @abstractmethod
    def get_value(self, model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
        """Return one bootstrap value for every environment instance."""

    @abstractmethod
    def build_optimizer(
        self,
        model: nn.Module,
        *,
        config: Mapping[str, Any],
    ) -> Optimizer:
        """Create the optimizer and its named parameter groups."""

    def update_state_stats(self, model: nn.Module, batch: Mapping[str, Any]) -> None:
        """Update optional observation normalization statistics."""

    def prepare_checkpoint_load(self, model: nn.Module) -> None:
        """Perform optional model-specific preparation before loading a checkpoint."""


class PlannerModelInterface(ABC):
    """Contract for an optional high-level planner used by a MARL method."""

    @property
    @abstractmethod
    def planner_name(self) -> str:
        """Return a stable planner name used in logs and checkpoints."""

    @abstractmethod
    def build_planner(
        self,
        *,
        agent_model: nn.Module,
        device: torch.device,
        config: Mapping[str, Any],
    ) -> nn.Module:
        """Construct the planner and connect it to the agent model."""

    @abstractmethod
    def plan(
        self,
        planner: nn.Module,
        batch: Mapping[str, Any],
        *,
        deterministic: bool = False,
    ) -> Mapping[str, Any]:
        """Generate high-level decisions or subtask conditions for each agent."""

    @abstractmethod
    def get_action_and_value(
        self,
        planner: nn.Module,
        batch: Mapping[str, Any],
        *,
        actions_input: Optional[Mapping[str, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[Mapping[str, Any], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], torch.Tensor]:
        """Return planner actions, log probabilities, entropies, and values."""

    @abstractmethod
    def build_optimizer(
        self,
        planner: nn.Module,
        *,
        config: Mapping[str, Any],
    ) -> Optimizer:
        """Create the planner optimizer."""

    @abstractmethod
    def update(
        self,
        planner: nn.Module,
        optimizer: Optimizer,
        rollout: Mapping[str, Any],
        *,
        config: Mapping[str, Any],
    ) -> Mapping[str, float]:
        """Apply one planner update and return scalar training metrics."""


class CommunicationInterface(ABC):
    """Contract for a method's inter-agent communication mechanism."""

    @property
    @abstractmethod
    def communication_name(self) -> str:
        """Return a stable communication mechanism name."""

    @abstractmethod
    def encode_message(
        self,
        sender: str,
        receiver: str,
        feature: torch.Tensor,
        *,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """Encode the information sent from one agent to another."""

    @abstractmethod
    def decode_message(
        self,
        message: Any,
        *,
        receiver: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Decode one received message into the receiver's feature space."""

    @abstractmethod
    def aggregate(
        self,
        local_feature: torch.Tensor,
        remote_features: Sequence[torch.Tensor],
        *,
        batch: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Fuse local and received features before policy inference."""

    def transmission_size(self, message: Any) -> int:
        """Return the serialized message size in bytes when measurable."""
        raise NotImplementedError


class TrainingLoopInterface(ABC):
    """Contract for connecting an agent, planner, and communication module to RL."""

    @abstractmethod
    def build_environment(
        self,
        *,
        workload: str,
        num_envs: int,
        config: Mapping[str, Any],
    ) -> Any:
        """Create the vectorized multi-agent environment."""

    @abstractmethod
    def collect_rollout(
        self,
        envs: Any,
        agent_model: nn.Module,
        *,
        planner: Optional[nn.Module] = None,
        communication: Optional[CommunicationInterface] = None,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Collect one multi-agent rollout and communication trace."""

    @abstractmethod
    def update(
        self,
        agent_model: nn.Module,
        optimizer: Optimizer,
        rollout: Mapping[str, Any],
        *,
        planner: Optional[nn.Module] = None,
        planner_optimizer: Optional[Optimizer] = None,
        config: Mapping[str, Any],
    ) -> Mapping[str, float]:
        """Perform one policy update and return scalar metrics."""

    @abstractmethod
    def evaluate(
        self,
        envs: Any,
        agent_model: nn.Module,
        *,
        planner: Optional[nn.Module] = None,
        episodes: int,
        config: Mapping[str, Any],
    ) -> Mapping[str, float]:
        """Evaluate task success and resource metrics."""

    @abstractmethod
    def save_checkpoint(
        self,
        path: Path,
        *,
        agent_model: nn.Module,
        planner: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        planner_optimizer: Optional[Optimizer] = None,
    ) -> None:
        """Save all method-specific states needed to resume training."""

    @abstractmethod
    def run(
        self,
        *,
        workload: str,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Run the complete training and evaluation workflow."""
