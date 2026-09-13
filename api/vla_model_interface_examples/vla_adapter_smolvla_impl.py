"""Workload-independent ``VLAModelInterface`` adapter for VLA-Adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from api.vla_model_interface import VLAAgentSpec, VLAModelInterface


class VLAAdapter(VLAModelInterface):
    """Expose VLA-Adapter without binding it to a specific workload."""

    def __init__(
        self,
        model_dir: str | Path,
        agent_spec: VLAAgentSpec,
        *,
        backend_module: str,
    ):
        self.model_dir = Path(model_dir)
        self._agent_spec = agent_spec
        self.backend_module = backend_module

    def _backend(self) -> ModuleType:
        """Load the selected workload's VLA-Adapter policy helpers lazily."""
        return importlib.import_module(self.backend_module)

    @property
    def model_name(self) -> str:
        return "vla_adapter_smolvla"

    @property
    def agent_spec(self) -> VLAAgentSpec:
        return self._agent_spec

    @property
    def policy_class(self) -> type[nn.Module]:
        return self._backend().MultiAgentVLAAdapterAgent

    def build_policy(self, *, device: torch.device, config: Mapping[str, Any]) -> nn.Module:
        state_dims = {self.agent_spec.state_dims[name] for name in self.agent_spec.agent_names}
        action_dims = {self.agent_spec.action_dims[name] for name in self.agent_spec.agent_names}
        if len(state_dims) != 1 or len(action_dims) != 1:
            raise ValueError("This shared VLA-Adapter example requires equal per-agent dimensions")
        policy = self.policy_class(
            agent_names=list(self.agent_spec.agent_names),
            state_dim=next(iter(state_dims)),
            global_state_dim=self.agent_spec.global_state_dim,
            action_dim=next(iter(action_dims)),
            model_dir=self.model_dir,
            normalize_state=bool(config.get("normalize_state", True)),
            freeze_vla_backbone=bool(config.get("freeze_vla_backbone", False)),
            image_size=config.get("image_size", 112),
            policy_mode=str(config.get("policy_mode", "native")),
            vision_token_pool_size=config.get("vision_token_pool_size", 16),
        )
        return policy.to(device)

    def build_batch_from_obs(self, obs: Any, *, device: torch.device) -> Mapping[str, Any]:
        batch = self._backend().build_batch_from_obs(obs, list(self.agent_spec.agent_names))
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def get_action_and_value(
        self,
        policy: nn.Module,
        batch: Mapping[str, Any],
        *,
        actions_input: Optional[Mapping[str, Any]] = None,
        deterministic: bool = False,
    ):
        del deterministic
        actions, log_probs, entropies, values = policy.get_action_and_value(
            dict(batch), actions_input=actions_input
        )
        return actions, log_probs, entropies, values

    def get_action(self, policy: nn.Module, batch: Mapping[str, Any], *, deterministic: bool = False):
        return policy.get_action(dict(batch), deterministic=deterministic)

    def get_value(self, policy: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
        return policy.get_value(dict(batch))

    def configure_trainable_modules(self, policy: nn.Module, *, freeze_vla_backbone: bool) -> None:
        policy.configure_trainable_modules(freeze_vla_backbone)

    def build_optimizer(self, policy: nn.Module, *, config: Mapping[str, Any]) -> Optimizer:
        class OptimizerConfig:
            pass

        args = OptimizerConfig()
        for key, value in config.items():
            setattr(args, key, value)
        return self._backend().build_optimizer(args, policy)
