"""Bind any MAPPO-compatible agent model to continual online RL."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from api.marl_method_interface import AgentModelInterface
from api.marl_online_rl_interface import MARLMethodSpec, MARLOnlineRLInterface
from train.marl.mappo.base import collect_rollout, mappo_update_on_policy


class MAPPOOnlineRL(MARLOnlineRLInterface):
    """Use a common agent-model implementation with MAPPO callbacks."""

    @property
    def method_spec(self) -> MARLMethodSpec:
        return MARLMethodSpec(
            algorithm_name="mappo",
            model_name="agent_model_mappo",
            collect_rollout_fn=collect_rollout,
            update_fn=mappo_update_on_policy,
        )

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
