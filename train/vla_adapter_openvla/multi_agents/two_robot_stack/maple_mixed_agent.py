import inspect
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from train.marl.maple.models import FutureStateHead, MultiAgentLatentCoordinator
from train.vla_adapter_openvla.multi_agents.two_robot_stack.mixed_sft_agent import (
    MixedTinyVLAAdapterOpenVLASFTAgent,
    build_mixed_mappo_optimizer,
)


def _zero_last_linear(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
            return


class OpenVLAMapleAdapterAgent(MixedTinyVLAAdapterOpenVLASFTAgent):
    def __init__(
        self,
        *args,
        latent_layers: int = 2,
        future_state_dim: Optional[int] = None,
        future_horizon: int = 1,
        **kwargs,
    ):
        parent_signature = inspect.signature(MixedTinyVLAAdapterOpenVLASFTAgent.__init__)
        parent_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in parent_signature.parameters
        }
        super().__init__(*args, **parent_kwargs)
        for parameter in self.parameters():
            parameter.requires_grad = False

        hidden_dim = int(self.vla_actor.hidden_dim)
        self.maple_hidden_dim = hidden_dim
        self.agent_feature_projectors = nn.ModuleDict(
            {
                self.vla_agent_name: nn.Sequential(
                    nn.LayerNorm(self.vla_actor.hidden_dim),
                    nn.Linear(self.vla_actor.hidden_dim, hidden_dim),
                ),
                self.smolvla_agent_name: nn.Sequential(
                    nn.LayerNorm(self.smolvla_actor.hidden_dim),
                    nn.Linear(self.smolvla_actor.hidden_dim, hidden_dim),
                ),
            }
        ).to(dtype=torch.float32)
        self.coordinator = MultiAgentLatentCoordinator(
            hidden_dim=hidden_dim,
            num_layers=latent_layers,
        ).to(dtype=torch.float32)
        self.agent_delta_heads = nn.ModuleDict(
            {
                self.vla_agent_name: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, self.vla_actor.hidden_dim),
                ),
                self.smolvla_agent_name: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, self.smolvla_actor.hidden_dim),
                ),
            }
        ).to(dtype=torch.float32)

        if future_state_dim is None:
            future_state_dim = self.global_state_dim
        self.future_state_dim = int(future_state_dim)
        self.future_horizon = int(future_horizon)
        self.future_state_head = FutureStateHead(
            input_dim=self.global_state_dim + hidden_dim * len(self.agent_names),
            hidden_dim=max(256, hidden_dim),
            future_state_dim=self.future_state_dim,
            horizon=self.future_horizon,
        ).to(dtype=torch.float32)

        for module in self.agent_delta_heads.values():
            _zero_last_linear(module)
        _zero_last_linear(self.future_state_head)

        for module in [
            self.agent_feature_projectors,
            self.coordinator,
            self.agent_delta_heads,
            self.future_state_head,
            self.critic_state_encoder,
            self.critic_visual_encoder,
            self.critic,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True

        self._latest_future_state = None

    def _adapt_actor_state_features(
        self,
        vla_state_feature: torch.Tensor,
        smolvla_state_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_tokens = torch.stack(
            [
                self.agent_feature_projectors[self.vla_agent_name](vla_state_feature),
                self.agent_feature_projectors[self.smolvla_agent_name](smolvla_state_feature),
            ],
            dim=1,
        )
        coordinated = self.coordinator(shared_tokens)
        delta_vla = self.agent_delta_heads[self.vla_agent_name](coordinated[:, 0, :])
        delta_smol = self.agent_delta_heads[self.smolvla_agent_name](coordinated[:, 1, :])
        global_state = getattr(self, "_maple_global_state_for_aux", None)
        if global_state is not None:
            future_input = torch.cat([global_state, coordinated.reshape(coordinated.shape[0], -1)], dim=-1)
            self._latest_future_state = self.future_state_head(future_input)
        return vla_state_feature + delta_vla, smolvla_state_feature + delta_smol

    def get_action_and_value(
        self,
        batch: Mapping[str, Any],
        actions_input: Optional[Mapping[str, Union[np.ndarray, torch.Tensor]]] = None,
        action_bins_input: Optional[Mapping[str, Union[np.ndarray, torch.Tensor]]] = None,
        return_action_bins: bool = False,
        return_aux: bool = False,
        deterministic: bool = False,
        return_token_logits: bool = False,
    ):
        global_state = batch["global_state"].to(device=self.device, dtype=torch.float32)
        self._maple_global_state_for_aux = self._normalize_global_state(global_state)
        self._latest_future_state = None
        try:
            result = super().get_action_and_value(
                batch,
                actions_input=actions_input,
                return_token_logits=return_token_logits,
                action_bins_input=action_bins_input,
                deterministic=deterministic,
            )
        finally:
            self._maple_global_state_for_aux = None

        if return_token_logits:
            actions_out, log_probs, entropies, value, token_logits = result
        else:
            actions_out, log_probs, entropies, value = result

        if return_action_bins:
            action_bins_out = {}
            if action_bins_input is not None:
                for name in self.agent_names:
                    if name in action_bins_input:
                        action_bins_out[name] = torch.as_tensor(
                            action_bins_input[name],
                            device=self.device,
                            dtype=torch.long,
                        )
            if self.vla_agent_name not in action_bins_out:
                if actions_input is not None and self.vla_agent_name in actions_input:
                    action_bins_out[self.vla_agent_name] = self.vla_actor.env_actions_to_bin_indices(
                        actions_input[self.vla_agent_name]
                    ).to(device=self.device, dtype=torch.long)
                else:
                    action_bins_out[self.vla_agent_name] = self.vla_actor.env_actions_to_bin_indices(
                        actions_out[self.vla_agent_name]
                    ).to(device=self.device, dtype=torch.long)
            if self.smolvla_agent_name not in action_bins_out:
                action_bins_out[self.smolvla_agent_name] = self._smolvla_actions_to_bin_indices(
                    actions_out[self.smolvla_agent_name]
                )
            return actions_out, log_probs, entropies, value, action_bins_out

        if return_aux:
            aux_outputs = {
                "future_state": self._latest_future_state,
                "joint_value": value,
            }
            return actions_out, log_probs, entropies, value, aux_outputs

        if return_token_logits:
            return actions_out, log_probs, entropies, value, token_logits
        return actions_out, log_probs, entropies, value

    @torch.no_grad()
    def get_action(self, batch: Mapping[str, Any], deterministic: bool = False) -> Dict[str, np.ndarray]:
        actions_out, _, _, _ = self.get_action_and_value(batch, deterministic=deterministic)
        return actions_out

    def get_value(self, batch: Mapping[str, Any]) -> torch.Tensor:
        _, _, _, value = self.get_action_and_value(batch, deterministic=True)
        return value

    @torch.no_grad()
    def predict_future_state(self, batch: Mapping[str, Any]) -> torch.Tensor:
        _, _, _, _, aux = self.get_action_and_value(batch, deterministic=True, return_aux=True)
        return aux["future_state"]


class OpenVLAMapleOnlineAgent(OpenVLAMapleAdapterAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for parameter in self.parameters():
            parameter.requires_grad = True


def build_optimizer(args, agent: OpenVLAMapleAdapterAgent) -> torch.optim.Optimizer:
    maple_param_groups = [
        {
            "params": [p for p in agent.agent_feature_projectors.parameters() if p.requires_grad]
            + [p for p in agent.coordinator.parameters() if p.requires_grad]
            + [p for p in agent.agent_delta_heads.parameters() if p.requires_grad],
            "lr": args.head_learning_rate,
            "group_name": "latent_coordination",
        },
        {
            "params": [p for p in agent.future_state_head.parameters() if p.requires_grad],
            "lr": getattr(args, "future_head_learning_rate", args.head_learning_rate),
            "group_name": "future_state_head",
        },
    ]
    maple_param_groups = [group for group in maple_param_groups if group["params"]]

    core_trainable = any(
        parameter.requires_grad
        for module in [
            agent.vla_actor.vla,
            agent.vla_actor.state_projector,
            agent.vla_actor.context_projector,
            agent.vla_actor.actor_head,
            agent.smolvla_actor.vla,
            agent.smolvla_actor.state_projector,
            agent.smolvla_actor.context_projector,
            agent.smolvla_actor.action_mean_head,
            agent.smolvla_actor.log_std_head,
            agent.critic_state_encoder,
            agent.critic_visual_encoder,
            agent.critic,
        ]
        for parameter in module.parameters()
    )

    if core_trainable:
        optimizer = build_mixed_mappo_optimizer(args, agent)
        for group in maple_param_groups:
            optimizer.add_param_group(group)
        return optimizer

    return torch.optim.AdamW(maple_param_groups, eps=1e-5, weight_decay=args.weight_decay)


SmolVLAMapleAdapterAgent = OpenVLAMapleAdapterAgent
SmolVLAMapleOnlineAgent = OpenVLAMapleOnlineAgent
