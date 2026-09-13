"""MAPLE interface example backed by the repository's multi-agent MAPLE agent."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.optim import Optimizer

from api.marl_method_interface import AgentModelInterface, CommunicationInterface
from api.marl_method_interface_examples.happo_impl import HAPPOTrainingLoop
from api.marl_method_interface_examples._common import MultiAgentActorCritic, TensorCommunication, batch_from_obs, validate_agent_spec


class LightweightMAPLEActorCritic(MultiAgentActorCritic):
    """Dependency-free latent rollout model for exercising MAPLE integration."""

    def __init__(self, agent_names: Sequence[str], state_dims: Mapping[str, int], action_dims: Mapping[str, int], hidden_dim: int = 128):
        super().__init__(agent_names, state_dims, action_dims, hidden_dim)
        total_state_dim = sum(state_dims.values())
        self.agent_token_encoder = nn.Sequential(nn.Linear(total_state_dim, hidden_dim), nn.Tanh())
        self.future_state_head = nn.Linear(hidden_dim, total_state_dim)

    def predict_future_state(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        states = torch.cat([batch[f"agent_states_{name}"] for name in self.agent_names], dim=-1)
        return self.future_state_head(self.agent_token_encoder(states))


class MAPLEAgentModel(AgentModelInterface):
    def __init__(self, agent_names: Sequence[str], state_dims: Mapping[str, int], action_dims: Mapping[str, int], model_dir: str | Path | None = None):
        self._agent_names = validate_agent_spec(agent_names, state_dims, action_dims)
        self.state_dims = dict(state_dims)
        self.action_dims = dict(action_dims)
        self.model_dir = Path(model_dir) if model_dir else None

    @property
    def model_name(self) -> str:
        return "maple_vla"

    @property
    def agent_names(self) -> Sequence[str]:
        return self._agent_names

    def build_policy(self, *, device: torch.device, config: Mapping[str, Any]) -> nn.Module:
        if self.model_dir is None:
            return LightweightMAPLEActorCritic(self.agent_names, self.state_dims, self.action_dims, int(config.get("hidden_dim", 128))).to(device)
        if len(set(self.state_dims.values())) != 1 or len(set(self.action_dims.values())) != 1:
            raise ValueError("the repository MAPLE agent requires equal per-agent dimensions")
        from train.marl.maple.models import MapleEdgeVLAAgent
        return MapleEdgeVLAAgent(
            agent_names=list(self.agent_names),
            state_dim=next(iter(self.state_dims.values())),
            global_state_dim=sum(self.state_dims.values()),
            action_dim=next(iter(self.action_dims.values())),
            model_dir=self.model_dir,
            normalize_state=bool(config.get("normalize_state", True)),
            freeze_vla_backbone=bool(config.get("freeze_vla_backbone", False)),
            image_size=config.get("image_size", 112),
            latent_layers=int(config.get("latent_layers", 2)),
        ).to(device)

    def build_batch_from_obs(self, obs: Any, *, device: torch.device) -> Mapping[str, Any]:
        if self.model_dir is not None:
            from train.internVL.model import build_batch_from_obs
            batch = build_batch_from_obs(obs, list(self.agent_names))
            return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        return batch_from_obs(obs, self.agent_names, self.state_dims, device=device)

    def get_action_and_value(self, model: nn.Module, batch: Mapping[str, Any], *, actions_input: Optional[Mapping[str, Any]] = None, deterministic: bool = False):
        return model.get_action_and_value(batch, actions_input=actions_input, deterministic=deterministic)

    def get_action(self, model: nn.Module, batch: Mapping[str, Any], *, deterministic: bool = False):
        return model.get_action(batch, deterministic=deterministic)

    def get_value(self, model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
        return model.get_value(batch)

    def build_optimizer(self, model: nn.Module, *, config: Mapping[str, Any]) -> Optimizer:
        if self.model_dir is None:
            return torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 3e-4)))
        from train.marl.maple.models import build_optimizer
        values = {"backbone_learning_rate": 1e-6, "state_learning_rate": 3e-6, "head_learning_rate": 3e-6, "value_head_learning_rate": 1e-4, "weight_decay": 1e-6, **config}
        return build_optimizer(SimpleNamespace(**values), model)


class MAPLECommunication(CommunicationInterface):
    communication_name = "maple_latent_tensor"

    def __init__(self) -> None:
        self._impl = TensorCommunication()

    def encode_message(self, sender: str, receiver: str, feature: torch.Tensor, *, action_mask: Optional[torch.Tensor] = None) -> Any:
        return self._impl.encode_message(sender, receiver, feature, action_mask=action_mask)

    def decode_message(self, message: Any, *, receiver: str, device: torch.device) -> torch.Tensor:
        return self._impl.decode_message(message, receiver=receiver, device=device)

    def aggregate(self, local_feature: torch.Tensor, remote_features: Sequence[torch.Tensor], *, batch: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
        return self._impl.aggregate(local_feature, remote_features, batch=batch)

    def transmission_size(self, message: Any) -> int:
        return self._impl.transmission_size(message)


class MAPLETrainingLoop(HAPPOTrainingLoop):
    """HAPPO-compatible policy update plus MAPLE latent-rollout supervision."""

    def update(self, agent_model: nn.Module, optimizer: Optimizer, rollout: Mapping[str, Any], *, planner: Optional[nn.Module] = None, planner_optimizer: Optional[Optimizer] = None, config: Mapping[str, Any]) -> Mapping[str, float]:
        metrics = super().update(
            agent_model,
            optimizer,
            rollout,
            planner=planner,
            planner_optimizer=planner_optimizer,
            config=config,
        )
        target = rollout.get("future_state_target")
        predictor = getattr(agent_model, "predict_future_state", None)
        coefficient = float(config.get("maple_future_state_coef", 1.0))
        if target is None or predictor is None or coefficient <= 0:
            metrics["future_state_loss"] = 0.0
            metrics["future_state_updated"] = 0.0
            return metrics
        prediction = predictor(rollout["batch"])
        target_tensor = torch.as_tensor(target, device=prediction.device, dtype=prediction.dtype)
        future_loss = torch.nn.functional.smooth_l1_loss(prediction, target_tensor)
        optimizer.zero_grad(set_to_none=True)
        (coefficient * future_loss).backward()
        nn.utils.clip_grad_norm_(agent_model.parameters(), float(config.get("max_grad_norm", 0.5)))
        optimizer.step()
        metrics["future_state_loss"] = float(future_loss.detach())
        metrics["future_state_updated"] = 1.0
        return metrics
