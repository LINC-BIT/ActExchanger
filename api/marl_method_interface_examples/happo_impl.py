"""HAPPO integration scaffold for ActExchanger.

The class boundaries are intentionally complete while the method bodies remain pending.
They document the integration points needed by the shared multi-agent runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.optim import Optimizer

from api.marl_method_interface import (
    AgentModelInterface,
    CommunicationInterface,
    PlannerModelInterface,
    TrainingLoopInterface,
)


class HAPPOAgentModel(AgentModelInterface):
    """CNN-based HAPPO agent model adapter."""

    def __init__(self, agent_names: Sequence[str], state_dims: Mapping[str, int], action_dims: Mapping[str, int]):
        self._agent_names = tuple(agent_names)
        self.state_dims = dict(state_dims)
        self.action_dims = dict(action_dims)

    @property
    def model_name(self) -> str:
        return "happo_cnn"

    @property
    def agent_names(self) -> Sequence[str]:
        return self._agent_names

    def build_model(self, *, device: torch.device, config: Mapping[str, Any]) -> nn.Module:
        raise NotImplementedError

    def build_batch_from_obs(self, obs: Any, *, device: torch.device) -> Mapping[str, Any]:
        raise NotImplementedError

    def get_action_and_value(self, model: nn.Module, batch: Mapping[str, Any], *, actions_input: Optional[Mapping[str, torch.Tensor]] = None, deterministic: bool = False):
        raise NotImplementedError

    def get_action(self, model: nn.Module, batch: Mapping[str, Any], *, deterministic: bool = False) -> Mapping[str, Any]:
        raise NotImplementedError

    def get_value(self, model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def build_optimizer(self, model: nn.Module, *, config: Mapping[str, Any]) -> Optimizer:
        raise NotImplementedError


class HAPPOPlannerModel(PlannerModelInterface):
    """Optional high-level planner adapter for HAPPO workloads."""

    @property
    def planner_name(self) -> str:
        return "happo_planner"

    def build_planner(self, *, agent_model: nn.Module, device: torch.device, config: Mapping[str, Any]) -> nn.Module:
        raise NotImplementedError

    def plan(self, planner: nn.Module, batch: Mapping[str, Any], *, deterministic: bool = False) -> Mapping[str, Any]:
        raise NotImplementedError

    def get_action_and_value(self, planner: nn.Module, batch: Mapping[str, Any], *, actions_input: Optional[Mapping[str, torch.Tensor]] = None, deterministic: bool = False):
        raise NotImplementedError

    def build_optimizer(self, planner: nn.Module, *, config: Mapping[str, Any]) -> Optimizer:
        raise NotImplementedError

    def update(self, planner: nn.Module, optimizer: Optimizer, rollout: Mapping[str, Any], *, config: Mapping[str, Any]) -> Mapping[str, float]:
        raise NotImplementedError


class HAPPOCommunication(CommunicationInterface):
    """Communication adapter for the HAPPO baseline."""

    @property
    def communication_name(self) -> str:
        return "happo_communication"

    def encode_message(self, sender: str, receiver: str, feature: torch.Tensor, *, action_mask: Optional[torch.Tensor] = None) -> Any:
        raise NotImplementedError

    def decode_message(self, message: Any, *, receiver: str, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    def aggregate(self, local_feature: torch.Tensor, remote_features: Sequence[torch.Tensor], *, batch: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
        raise NotImplementedError


class HAPPOTrainingLoop(TrainingLoopInterface):
    """Training-loop adapter for the four multi-agent workloads."""

    def build_environment(self, *, workload: str, num_envs: int, config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def collect_rollout(self, envs: Any, agent_model: nn.Module, *, planner: Optional[nn.Module] = None, communication: Optional[CommunicationInterface] = None, config: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError

    def update(self, agent_model: nn.Module, optimizer: Optimizer, rollout: Mapping[str, Any], *, planner: Optional[nn.Module] = None, planner_optimizer: Optional[Optimizer] = None, config: Mapping[str, Any]) -> Mapping[str, float]:
        raise NotImplementedError

    def evaluate(self, envs: Any, agent_model: nn.Module, *, planner: Optional[nn.Module] = None, episodes: int, config: Mapping[str, Any]) -> Mapping[str, float]:
        raise NotImplementedError

    def save_checkpoint(self, path: Path, *, agent_model: nn.Module, planner: Optional[nn.Module] = None, optimizer: Optional[Optimizer] = None, planner_optimizer: Optional[Optimizer] = None) -> None:
        raise NotImplementedError

    def run(self, *, workload: str, config: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError
