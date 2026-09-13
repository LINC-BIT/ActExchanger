"""Small runnable components shared by the MARL interface examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.optim import Optimizer


def validate_agent_spec(
    agent_names: Sequence[str],
    state_dims: Mapping[str, int],
    action_dims: Mapping[str, int],
) -> tuple[str, ...]:
    """Validate the dimension maps shared by the runnable examples."""
    names = tuple(agent_names)
    if not names:
        raise ValueError("agent_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("agent_names must be unique")
    for name in names:
        if int(state_dims.get(name, 0)) <= 0:
            raise ValueError(f"state_dims[{name!r}] must be positive")
        if int(action_dims.get(name, 0)) <= 0:
            raise ValueError(f"action_dims[{name!r}] must be positive")
    return names


def _as_state(value: Any, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    state = torch.as_tensor(value, device=device, dtype=dtype)
    return state.unsqueeze(0) if state.ndim == 1 else state


def batch_from_obs(
    obs: Any,
    agent_names: Sequence[str],
    state_dims: Mapping[str, int],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not isinstance(obs, Mapping):
        raise TypeError("multi-agent observations must be a mapping")
    batch: dict[str, torch.Tensor] = {}
    states = []
    for name in agent_names:
        key = f"agent_states_{name}"
        value = obs.get(key)
        if value is None:
            value = obs.get(name)
            if isinstance(value, Mapping):
                value = value.get("state", value.get("states"))
        if value is None:
            raise KeyError(f"missing state for agent {name!r}")
        state = _as_state(value, device=device)
        if state.shape[-1] != int(state_dims[name]):
            raise ValueError(f"state dimension for {name!r} is {state.shape[-1]}, expected {state_dims[name]}")
        batch[key] = state
        states.append(state)
    global_state = obs.get("global_state")
    if global_state is None:
        global_state = torch.cat(states, dim=-1)
    batch["global_state"] = _as_state(global_state, device=device)
    return batch


class MultiAgentActorCritic(nn.Module):
    def __init__(self, agent_names: Sequence[str], state_dims: Mapping[str, int], action_dims: Mapping[str, int], hidden_dim: int = 128):
        super().__init__()
        self.agent_names = tuple(agent_names)
        self.state_dims = dict(state_dims)
        self.action_dims = dict(action_dims)
        self.actors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.state_dims[name], hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, self.action_dims[name]),
            )
            for name in self.agent_names
        })
        total_state_dim = sum(self.state_dims.values())
        self.critic = nn.Sequential(
            nn.Linear(total_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_stds = nn.ParameterDict({name: nn.Parameter(torch.zeros(self.action_dims[name])) for name in self.agent_names})

    def get_action_and_value(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        actions_input: Optional[Mapping[str, torch.Tensor]] = None,
        deterministic: bool = False,
    ):
        actions: dict[str, torch.Tensor] = {}
        log_probs: dict[str, torch.Tensor] = {}
        entropies: dict[str, torch.Tensor] = {}
        states = []
        for name in self.agent_names:
            state = batch[f"agent_states_{name}"]
            states.append(state)
            mean = self.actors[name](state)
            distribution = torch.distributions.Normal(mean, self.log_stds[name].exp().expand_as(mean))
            action = actions_input[name] if actions_input is not None and name in actions_input else (mean if deterministic else distribution.rsample())
            actions[name] = action
            log_probs[name] = distribution.log_prob(action).sum(dim=-1)
            entropies[name] = distribution.entropy().sum(dim=-1)
        value = self.critic(torch.cat(states, dim=-1)).squeeze(-1)
        return actions, log_probs, entropies, value

    @torch.no_grad()
    def get_action(self, batch: Mapping[str, torch.Tensor], *, deterministic: bool = False):
        return self.get_action_and_value(batch, deterministic=deterministic)[0]

    def get_value(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(torch.cat([batch[f"agent_states_{name}"] for name in self.agent_names], dim=-1)).squeeze(-1)


class ZeroPlanner(nn.Module):
    def __init__(self, agent_names: Sequence[str]):
        super().__init__()
        self.agent_names = tuple(agent_names)
        self.offset = nn.Parameter(torch.zeros(1))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        size = batch[f"agent_states_{self.agent_names[0]}"].shape[0]
        return {name: self.offset.expand(size, 1) for name in self.agent_names}


class TensorCommunication:
    communication_name = "tensor_mean"

    def encode_message(self, sender: str, receiver: str, feature: torch.Tensor, *, action_mask: Optional[torch.Tensor] = None) -> dict[str, Any]:
        return {"sender": sender, "receiver": receiver, "feature": feature.detach().clone(), "action_mask": action_mask}

    def decode_message(self, message: Mapping[str, Any], *, receiver: str, device: torch.device) -> torch.Tensor:
        if message["receiver"] != receiver:
            raise ValueError(f"message is addressed to {message['receiver']!r}, not {receiver!r}")
        return message["feature"].to(device)

    def aggregate(self, local_feature: torch.Tensor, remote_features: Sequence[torch.Tensor], *, batch: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
        del batch
        if not remote_features:
            return local_feature
        if any(feature.shape != local_feature.shape for feature in remote_features):
            raise ValueError("all communication features must have the same shape")
        return torch.stack([local_feature, *remote_features], dim=0).mean(dim=0)

    def transmission_size(self, message: Mapping[str, Any]) -> int:
        feature = message.get("feature")
        mask = message.get("action_mask")
        size = feature.numel() * feature.element_size() if torch.is_tensor(feature) else 0
        if torch.is_tensor(mask):
            size += mask.numel() * mask.element_size()
        return int(size)


class ReferenceTrainingLoop:
    """Small but functional loop used by the three integration examples.

    A real workload supplies ``environment_factory`` and an agent interface in the
    config. Keeping this orchestration here makes the examples executable without
    importing ManiSkill during unit and API smoke tests.
    """

    def build_environment(self, *, workload: str, num_envs: int, config: Mapping[str, Any]) -> Any:
        factory = config.get("environment_factory")
        if factory is None:
            raise ValueError("config['environment_factory'] is required")
        return factory(workload=workload, num_envs=int(num_envs), config=config)

    def collect_rollout(self, envs: Mapping[str, Any], agent_model: nn.Module, *, planner: Optional[nn.Module] = None, communication: Optional[TensorCommunication] = None, config: Mapping[str, Any]) -> Mapping[str, Any]:
        collector = config.get("rollout_collector")
        if collector is not None:
            return collector(envs, agent_model, planner=planner, communication=communication, config=config)
        collect = getattr(envs, "collect_rollout", None)
        if collect is None:
            raise TypeError("environment must implement collect_rollout(), or config must provide rollout_collector")
        return collect(agent_model, planner=planner, communication=communication, config=config)

    def update(self, agent_model: nn.Module, optimizer: Optimizer, rollout: Mapping[str, Any], *, planner: Optional[nn.Module] = None, planner_optimizer: Optional[Optimizer] = None, config: Mapping[str, Any]) -> Mapping[str, float]:
        del planner, planner_optimizer
        required = {"batch", "actions", "old_log_probs", "advantages", "returns"}
        missing = required.difference(rollout)
        if missing:
            raise KeyError(f"rollout is missing: {', '.join(sorted(missing))}")
        _, log_probs, entropies, values = agent_model.get_action_and_value(
            rollout["batch"], actions_input=rollout["actions"]
        )
        advantages = torch.as_tensor(rollout["advantages"], device=values.device, dtype=values.dtype)
        returns = torch.as_tensor(rollout["returns"], device=values.device, dtype=values.dtype)
        clip_coef = float(config.get("clip_coef", 0.2))
        entropy_coef = float(config.get("entropy_coef", 0.01))
        value_coef = float(config.get("value_coef", 0.5))
        policy_losses = []
        entropy_terms = []
        for name, new_log_prob in log_probs.items():
            old_log_prob = torch.as_tensor(rollout["old_log_probs"][name], device=values.device).detach()
            ratio = (new_log_prob - old_log_prob).exp()
            policy_losses.append(-torch.min(ratio * advantages, ratio.clamp(1 - clip_coef, 1 + clip_coef) * advantages).mean())
            entropy_terms.append(entropies[name].mean())
        policy_loss = torch.stack(policy_losses).mean()
        entropy = torch.stack(entropy_terms).mean()
        value_loss = 0.5 * (values - returns).pow(2).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(agent_model.parameters(), float(config.get("max_grad_norm", 0.5)))
        optimizer.step()
        return {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
        }

    def evaluate(self, envs: Mapping[str, Any], agent_model: nn.Module, *, planner: Optional[nn.Module] = None, episodes: int, config: Mapping[str, Any]) -> Mapping[str, float]:
        evaluator = config.get("evaluator")
        if evaluator is not None:
            return evaluator(envs, agent_model, planner=planner, episodes=episodes, config=config)
        evaluate = getattr(envs, "evaluate", None)
        if evaluate is None:
            raise TypeError("environment must implement evaluate(), or config must provide evaluator")
        return evaluate(agent_model, planner=planner, episodes=episodes, config=config)

    def save_checkpoint(self, path: Path, *, agent_model: nn.Module, planner: Optional[nn.Module] = None, optimizer: Optional[Optimizer] = None, planner_optimizer: Optional[Optimizer] = None) -> None:
        payload = {"agent_model": agent_model.state_dict()}
        if planner is not None:
            payload["planner"] = planner.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if planner_optimizer is not None:
            payload["planner_optimizer"] = planner_optimizer.state_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def run(self, *, workload: str, config: Mapping[str, Any]) -> Mapping[str, Any]:
        agent_impl = config.get("agent_model_interface")
        if agent_impl is None:
            raise ValueError("config['agent_model_interface'] is required")
        device = torch.device(config.get("device", "cpu"))
        model = agent_impl.build_policy(device=device, config=config)
        optimizer = agent_impl.build_optimizer(model, config=config)
        planner_impl = config.get("planner_model_interface")
        planner = None
        planner_optimizer = None
        if planner_impl is not None:
            planner = planner_impl.build_planner(agent_model=model, device=device, config=config)
            planner_optimizer = planner_impl.build_optimizer(planner, config=config)
        communication = config.get("communication_interface")
        envs = self.build_environment(
            workload=workload,
            num_envs=int(config.get("num_envs", 1)),
            config=config,
        )
        updates = max(1, int(config.get("updates", 1)))
        metrics: Mapping[str, float] = {}
        for _ in range(updates):
            rollout = self.collect_rollout(
                envs,
                model,
                planner=planner,
                communication=communication,
                config=config,
            )
            metrics = self.update(
                model,
                optimizer,
                rollout,
                planner=planner,
                planner_optimizer=planner_optimizer,
                config=config,
            )
        evaluation = self.evaluate(
            envs,
            model,
            episodes=int(config.get("evaluation_episodes", 1)),
            config=config,
        )
        checkpoint_path = config.get("checkpoint_path")
        if checkpoint_path is not None:
            self.save_checkpoint(
                Path(checkpoint_path),
                agent_model=model,
                planner=planner,
                optimizer=optimizer,
                planner_optimizer=planner_optimizer,
            )
        return {"workload": workload, "updates": updates, "training": dict(metrics), "evaluation": dict(evaluation)}
