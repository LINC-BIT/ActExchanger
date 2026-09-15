"""Scaffold for integrating a new multi-agent VLA model."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from api.vla_model_interface import VLAActionOutput, VLAAgentSpec, VLAModelInterface


class NewVLAModelImplementation(VLAModelInterface):
    """Fill in this adapter to expose a new VLA model to ActExchanger."""

    def __init__(self, agent_spec: VLAAgentSpec):
        self._agent_spec = agent_spec

    @property
    def model_name(self) -> str:
        return "new_vla_model"

    @property
    def agent_spec(self) -> VLAAgentSpec:
        return self._agent_spec

    @property
    def policy_class(self) -> type[nn.Module]:
        raise NotImplementedError

    def build_policy(self, *, device: torch.device, config: Mapping[str, Any]) -> nn.Module:
        raise NotImplementedError

    def build_batch_from_obs(self, obs: Any, *, device: torch.device) -> Mapping[str, Any]:
        raise NotImplementedError

    def generate_actions(
        self,
        policy: nn.Module,
        batch: Mapping[str, Any],
        *,
        actions_input: Optional[Mapping[str, Any]] = None,
        deterministic: bool = False,
        return_value: bool = False,
        generation_config: Optional[Mapping[str, Any]] = None,
    ) -> VLAActionOutput:
        raise NotImplementedError

    def get_value(self, policy: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def configure_trainable_modules(
        self,
        policy: nn.Module,
        *,
        freeze_vla_backbone: bool,
    ) -> None:
        raise NotImplementedError

    def build_optimizer(self, policy: nn.Module, *, config: Mapping[str, Any]) -> Optimizer:
        raise NotImplementedError
