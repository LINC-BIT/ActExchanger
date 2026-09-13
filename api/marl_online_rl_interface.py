"""Bind a common agent-model interface to a MARL online-RL method."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from api.marl_method_interface import AgentModelInterface


@dataclass(frozen=True)
class MARLMethodSpec:
    """Method callbacks consumed by the shared continual online-RL driver."""

    algorithm_name: str
    model_name: str
    collect_rollout_fn: Callable[..., Any]
    update_fn: Callable[..., Mapping[str, float]]
    rollout_mode: str = "mappo"
    train_mode_during_update: bool = False

    def __post_init__(self) -> None:
        if not self.algorithm_name.strip() or not self.model_name.strip():
            raise ValueError("algorithm_name and model_name must not be empty")
        if self.rollout_mode not in {"mappo", "maple"}:
            raise ValueError("rollout_mode must be 'mappo' or 'maple'")


class MARLOnlineRLInterface(ABC):
    """Method-specific adapter that runs an agent model with online RL.

    Subclasses select one MARL rollout/update pair and wrap the policy where that
    MARL method requires additional heads, latent variables, or a hierarchical planner.
    The existing driver retains continual environment switching, evaluation, and
    checkpoint scheduling.
    """

    @property
    @abstractmethod
    def method_spec(self) -> MARLMethodSpec:
        """Return the MARL algorithm and its rollout/update callbacks."""

    @abstractmethod
    def build_agent(
        self,
        *,
        args: Any,
        infos: Mapping[str, Any],
        agent_model: AgentModelInterface,
        device: torch.device,
    ) -> nn.Module:
        """Build an agent compatible with the selected MARL method."""

    def build_optimizer(
        self,
        *,
        args: Any,
        agent: nn.Module,
        agent_model: AgentModelInterface,
    ) -> Optimizer:
        """Build the optimizer through the common agent-model interface."""
        return agent_model.build_optimizer(agent, config=vars(args))

    def make_collate_fn(
        self,
        *,
        agent_names: list[str],
        agent_model: AgentModelInterface,
        device: torch.device,
    ) -> Callable[[Any], Mapping[str, Any]]:
        """Build the rollout collation function for the model's observation schema."""
        del agent_names

        def collate_fn(obs: Any) -> Mapping[str, Any]:
            return agent_model.build_batch_from_obs(obs, device=device)

        return collate_fn

    def make_sample_fn(
        self,
        *,
        agent_names: list[str],
        agent: nn.Module,
        agent_model: AgentModelInterface,
        device: torch.device,
        deterministic: bool,
    ) -> Callable[[Any], Mapping[str, Any]]:
        """Build the evaluation action function with the same batch conversion."""
        del agent_names

        def sample_fn(obs: Any) -> Mapping[str, Any]:
            batch = agent_model.build_batch_from_obs(obs, device=device)
            return agent_model.get_action(agent, batch, deterministic=deterministic)

        return sample_fn


class CallbackMARLOnlineRL(MARLOnlineRLInterface):
    """Connect driver-compatible MARL callbacks to any agent-model interface."""

    def __init__(
        self,
        *,
        algorithm_name: str,
        model_name: str,
        collect_rollout_fn: Callable[..., Any],
        update_fn: Callable[..., Mapping[str, float]],
        rollout_mode: str = "mappo",
        train_mode_during_update: bool = False,
    ) -> None:
        self._method_spec = MARLMethodSpec(
            algorithm_name=algorithm_name,
            model_name=model_name,
            collect_rollout_fn=collect_rollout_fn,
            update_fn=update_fn,
            rollout_mode=rollout_mode,
            train_mode_during_update=train_mode_during_update,
        )

    @property
    def method_spec(self) -> MARLMethodSpec:
        return self._method_spec

    def build_agent(
        self,
        *,
        args: Any,
        infos: Mapping[str, Any],
        agent_model: AgentModelInterface,
        device: torch.device,
    ) -> nn.Module:
        return agent_model.build_policy(
            device=device,
            config={**vars(args), **infos},
        )


def run_continual_online_rl(
    args: Any,
    *,
    agent_model: AgentModelInterface,
    marl_method: MARLOnlineRLInterface,
    driver: Callable[..., Any],
    env_kwargs_list: Optional[list[Mapping[str, Any]]] = None,
    init_label: Optional[str] = None,
) -> Any:
    """Run an agent model with a MARL method and continual environment changes.

    ``driver`` should be a workload's ``run_online_training`` function. It keeps the
    existing ``--env-change-time-points`` scheduling behavior while callbacks below
    supply the model-specific policy and batch schema.
    """
    method_spec = marl_method.method_spec

    def build_agent(runtime_args: Any, infos: Mapping[str, Any], *, device: torch.device) -> nn.Module:
        return marl_method.build_agent(
            args=runtime_args,
            infos=infos,
            agent_model=agent_model,
            device=device,
        )

    def build_optimizer(runtime_args: Any, agent: nn.Module) -> Optimizer:
        return marl_method.build_optimizer(
            args=runtime_args,
            agent=agent,
            agent_model=agent_model,
        )

    def collate_fn_builder(agent_names: list[str], device: torch.device):
        return marl_method.make_collate_fn(
            agent_names=agent_names,
            agent_model=agent_model,
            device=device,
        )

    def sample_fn_builder(
        agent_names: list[str],
        agent: nn.Module,
        device: torch.device,
        *,
        deterministic: bool,
    ):
        return marl_method.make_sample_fn(
            agent_names=agent_names,
            agent=agent,
            agent_model=agent_model,
            device=device,
            deterministic=deterministic,
        )

    return driver(
        args,
        algo_name=method_spec.algorithm_name,
        model_name=method_spec.model_name,
        build_agent=build_agent,
        build_optimizer=build_optimizer,
        update_fn=method_spec.update_fn,
        collect_rollout_fn=method_spec.collect_rollout_fn,
        rollout_mode=method_spec.rollout_mode,
        train_mode_during_update=method_spec.train_mode_during_update,
        env_kwargs_list=env_kwargs_list,
        init_label=init_label or f"{method_spec.algorithm_name} online init",
        collate_fn_builder=collate_fn_builder,
        sample_fn_builder=sample_fn_builder,
    )
