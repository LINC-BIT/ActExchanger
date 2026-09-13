"""Runnable, lightweight HAPPO-style multi-agent interface example."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.optim import Optimizer

from api.marl_method_interface import AgentModelInterface, CommunicationInterface, PlannerModelInterface, TrainingLoopInterface
from api.marl_method_interface_examples._common import MultiAgentActorCritic, ReferenceTrainingLoop, TensorCommunication, ZeroPlanner, batch_from_obs, validate_agent_spec


class HAPPOAgentModel(AgentModelInterface):
    def __init__(self, agent_names: Sequence[str], state_dims: Mapping[str, int], action_dims: Mapping[str, int]):
        self._agent_names = validate_agent_spec(agent_names, state_dims, action_dims)
        self.state_dims = dict(state_dims)
        self.action_dims = dict(action_dims)

    @property
    def model_name(self) -> str:
        return "happo_cnn"

    @property
    def agent_names(self) -> Sequence[str]:
        return self._agent_names

    def build_policy(self, *, device: torch.device, config: Mapping[str, Any]) -> nn.Module:
        return MultiAgentActorCritic(self.agent_names, self.state_dims, self.action_dims, int(config.get("hidden_dim", 128))).to(device)

    def build_batch_from_obs(self, obs: Any, *, device: torch.device) -> Mapping[str, Any]:
        return batch_from_obs(obs, self.agent_names, self.state_dims, device=device)

    def get_action_and_value(self, model: nn.Module, batch: Mapping[str, Any], *, actions_input: Optional[Mapping[str, torch.Tensor]] = None, deterministic: bool = False):
        return model.get_action_and_value(batch, actions_input=actions_input, deterministic=deterministic)

    def get_action(self, model: nn.Module, batch: Mapping[str, Any], *, deterministic: bool = False):
        return model.get_action(batch, deterministic=deterministic)

    def get_value(self, model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
        return model.get_value(batch)

    def build_optimizer(self, model: nn.Module, *, config: Mapping[str, Any]) -> Optimizer:
        return torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 3e-4)))


class HAPPOPlannerModel(PlannerModelInterface):
    planner_name = "happo_planner"

    def build_planner(self, *, agent_model: nn.Module, device: torch.device, config: Mapping[str, Any]) -> nn.Module:
        del config
        return ZeroPlanner(agent_model.agent_names).to(device)

    def plan(self, planner: nn.Module, batch: Mapping[str, Any], *, deterministic: bool = False) -> Mapping[str, Any]:
        del deterministic
        return planner(batch)

    def get_action_and_value(self, planner: nn.Module, batch: Mapping[str, Any], *, actions_input: Optional[Mapping[str, torch.Tensor]] = None, deterministic: bool = False):
        del actions_input, deterministic
        output = planner(batch)
        values = torch.zeros(next(iter(output.values())).shape[0], device=next(iter(output.values())).device)
        return output, {}, {}, values

    def build_optimizer(self, planner: nn.Module, *, config: Mapping[str, Any]) -> Optimizer:
        return torch.optim.Adam(planner.parameters(), lr=float(config.get("learning_rate", 3e-4)))

    def update(self, planner: nn.Module, optimizer: Optimizer, rollout: Mapping[str, Any], *, config: Mapping[str, Any]) -> Mapping[str, float]:
        del config
        target = rollout.get("planner_targets")
        batch = rollout.get("batch")
        if target is None or batch is None:
            return {"planner_loss": 0.0, "planner_updated": 0.0}
        output = planner(batch)
        losses = [torch.nn.functional.mse_loss(output[name], target[name].to(output[name])) for name in output]
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return {"planner_loss": float(loss.detach()), "planner_updated": 1.0}


class HAPPOCommunication(CommunicationInterface):
    communication_name = "happo_tensor_mean"

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


class HAPPOTrainingLoop(ReferenceTrainingLoop, TrainingLoopInterface):
    """HAPPO sequential PPO update over independent per-agent actors."""

    def update(self, agent_model: nn.Module, optimizer: Optimizer, rollout: Mapping[str, Any], *, planner: Optional[nn.Module] = None, planner_optimizer: Optional[Optimizer] = None, config: Mapping[str, Any]) -> Mapping[str, float]:
        required = {"batch", "actions", "old_log_probs", "advantages", "returns"}
        missing = required.difference(rollout)
        if missing:
            raise KeyError(f"rollout is missing: {', '.join(sorted(missing))}")
        batch = rollout["batch"]
        actions = rollout["actions"]
        with torch.no_grad():
            _, _, _, initial_values = agent_model.get_action_and_value(batch, actions_input=actions)
        advantages = torch.as_tensor(rollout["advantages"], device=initial_values.device, dtype=initial_values.dtype)
        returns = torch.as_tensor(rollout["returns"], device=initial_values.device, dtype=initial_values.dtype)
        if bool(config.get("normalize_advantages", True)) and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        clip_coef = float(config.get("clip_coef", 0.2))
        entropy_coef = float(config.get("entropy_coef", 0.01))
        value_coef = float(config.get("value_coef", 0.5))
        max_grad_norm = float(config.get("max_grad_norm", 0.5))
        order = list(config.get("agent_update_order", agent_model.agent_names))
        if set(order) != set(agent_model.agent_names) or len(order) != len(agent_model.agent_names):
            raise ValueError("agent_update_order must contain every agent exactly once")
        factor = torch.ones_like(advantages)
        actor_losses: dict[str, float] = {}
        for name in order:
            old_log_prob = torch.as_tensor(rollout["old_log_probs"][name], device=advantages.device).detach()
            _, log_probs, entropies, _ = agent_model.get_action_and_value(batch, actions_input=actions)
            ratio = (log_probs[name] - old_log_prob).exp()
            surrogate = torch.min(ratio * advantages, ratio.clamp(1 - clip_coef, 1 + clip_coef) * advantages)
            actor_loss = -(factor.detach() * surrogate).mean() - entropy_coef * entropies[name].mean()
            optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            parameters = [*agent_model.actors[name].parameters(), agent_model.log_stds[name]]
            nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            # Clear gradients of other actors and the central critic. HAPPO changes
            # exactly one heterogeneous policy at each step.
            allowed = {id(parameter) for parameter in parameters}
            for parameter in agent_model.parameters():
                if id(parameter) not in allowed:
                    parameter.grad = None
            optimizer.step()
            with torch.no_grad():
                _, updated_log_probs, _, _ = agent_model.get_action_and_value(batch, actions_input=actions)
                factor = factor * (updated_log_probs[name] - old_log_prob).exp()
            actor_losses[name] = float(actor_loss.detach())
        values = agent_model.get_value(batch)
        value_loss = 0.5 * (values - returns).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        (value_coef * value_loss).backward()
        nn.utils.clip_grad_norm_(agent_model.critic.parameters(), max_grad_norm)
        optimizer.step()
        metrics: dict[str, float] = {
            "loss": float(sum(actor_losses.values()) / len(actor_losses) + value_coef * value_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "importance_factor": float(factor.mean()),
        }
        metrics.update({f"policy_loss/{name}": loss for name, loss in actor_losses.items()})
        if planner is not None and planner_optimizer is not None:
            metrics.update(HAPPOPlannerModel().update(planner, planner_optimizer, rollout, config=config))
        return metrics
