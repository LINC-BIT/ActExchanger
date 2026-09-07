from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from train.marl.maple.models import FutureStateHead, MultiAgentLatentCoordinator
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.toy_cnn_agent import (
    PassObjectToyCNNAgent,
    build_toy_cnn_mappo_optimizer,
)


def _zero_last_linear(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
            return


class FiveVLAMapleAdapterAgent(PassObjectToyCNNAgent):
    ACTOR_MEAN_CLAMP = 20.0
    ACTOR_LOGSTD_MIN = -5.0
    ACTOR_LOGSTD_MAX = 2.0

    def __init__(
        self,
        *args,
        latent_layers: int = 2,
        future_state_dim: Optional[int] = None,
        future_horizon: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        for parameter in self.parameters():
            parameter.requires_grad = False

        hidden_dim = 256
        self.maple_hidden_dim = hidden_dim
        self.agent_feature_projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for name in self.agent_names
            }
        ).to(dtype=torch.float32)
        self.coordinator = MultiAgentLatentCoordinator(
            hidden_dim=hidden_dim,
            num_layers=latent_layers,
        ).to(dtype=torch.float32)
        self.agent_delta_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for name in self.agent_names
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
            self.rgb_encoder,
            self.critic_state_encoder,
            self.critic,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True

        self._latest_future_state = None

    def _adapt_actor_state_features(
        self,
        state_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        shared_tokens = torch.stack(
            [self.agent_feature_projectors[name](state_features[name]) for name in self.agent_names],
            dim=1,
        )
        coordinated = self.coordinator(shared_tokens)
        global_state = getattr(self, "_maple_global_state_for_aux", None)
        if global_state is not None:
            future_input = torch.cat([global_state, coordinated.reshape(coordinated.shape[0], -1)], dim=-1)
            self._latest_future_state = self.future_state_head(future_input)
        return {
            name: state_features[name] + self.agent_delta_heads[name](coordinated[:, idx, :])
            for idx, name in enumerate(self.agent_names)
        }

    def _safe_distribution(self, mean: torch.Tensor, logstd: torch.Tensor):
        mean = torch.nan_to_num(
            mean.float(),
            nan=0.0,
            posinf=self.ACTOR_MEAN_CLAMP,
            neginf=-self.ACTOR_MEAN_CLAMP,
        ).clamp(-self.ACTOR_MEAN_CLAMP, self.ACTOR_MEAN_CLAMP)
        logstd = torch.nan_to_num(
            logstd.float(),
            nan=0.0,
            posinf=self.ACTOR_LOGSTD_MAX,
            neginf=self.ACTOR_LOGSTD_MIN,
        ).clamp(self.ACTOR_LOGSTD_MIN, self.ACTOR_LOGSTD_MAX)
        std = torch.exp(logstd)
        return mean, std

    def _safe_action(self, action, *, mean_ref: torch.Tensor):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=mean_ref.dtype, device=mean_ref.device)
        else:
            action = action.to(device=mean_ref.device, dtype=mean_ref.dtype)
        return torch.nan_to_num(
            action.float(),
            nan=0.0,
            posinf=self.ACTOR_MEAN_CLAMP,
            neginf=-self.ACTOR_MEAN_CLAMP,
        )

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
        batch = self._prepare_batch(batch)
        global_state = batch["global_state"].to(device=self.device, dtype=torch.float32)
        self._latest_future_state = None
        self._maple_global_state_for_aux = self._normalize_global_state(global_state)
        try:
            rgb = batch["rgb"]
            rgb_feat = self.rgb_encoder(rgb)
            critic_feat = self._critic_feature(rgb_feat, self._normalize_global_state(global_state))
            value = self.critic(critic_feat).squeeze(-1)

            base_state_features = {}
            for name in self.agent_names:
                agent_state = self._normalize_actor_state(name, batch[f"agent_states_{name}"])
                base_state_features[name] = self.actor_state_encoders[name](agent_state)
            adapted_state_features = self._adapt_actor_state_features(base_state_features)

            actions_out = {}
            log_probs = {}
            entropies = {}
            for name in self.agent_names:
                actor_feat = torch.cat([rgb_feat, adapted_state_features[name]], dim=1)
                mean = self.actor_heads[name](actor_feat)
                logstd = self.actor_logstd[name].expand_as(mean)
                mean, std = self._safe_distribution(mean, logstd)
                dist = torch.distributions.Normal(mean, std)

                if action_bins_input is not None and name in action_bins_input:
                    action = action_bins_input[name]
                elif actions_input is not None and name in actions_input:
                    action = actions_input[name]
                else:
                    action = mean if deterministic else dist.sample()

                action = self._safe_action(action, mean_ref=mean)

                actions_out[name] = action.detach().cpu().numpy()
                log_probs[name] = dist.log_prob(action).sum(-1)
                entropies[name] = dist.entropy().sum(-1)

            future_input = torch.cat(
                [self._normalize_global_state(global_state), rgb_feat.new_zeros(rgb_feat.shape[0], self.maple_hidden_dim * len(self.agent_names))],
                dim=-1,
            )
            self._latest_future_state = self.future_state_head(future_input)

            if return_aux:
                aux_outputs = {
                    "future_state": self._latest_future_state,
                    "joint_value": value,
                }
                if return_action_bins:
                    return actions_out, log_probs, entropies, value, actions_out, aux_outputs
                if return_token_logits:
                    return actions_out, log_probs, entropies, value, None, aux_outputs
                return actions_out, log_probs, entropies, value, aux_outputs

            if return_action_bins and return_token_logits:
                return actions_out, log_probs, entropies, value, actions_out, None
            if return_action_bins:
                return actions_out, log_probs, entropies, value, actions_out
            if return_token_logits:
                return actions_out, log_probs, entropies, value, None
            return actions_out, log_probs, entropies, value
        finally:
            self._maple_global_state_for_aux = None

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


ThreeVLAMapleAdapterAgent = FiveVLAMapleAdapterAgent


def build_optimizer(args, agent: FiveVLAMapleAdapterAgent) -> torch.optim.Optimizer:
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
    optimizer = build_toy_cnn_mappo_optimizer(args, agent)
    for group in maple_param_groups:
        optimizer.add_param_group(group)
    return optimizer
