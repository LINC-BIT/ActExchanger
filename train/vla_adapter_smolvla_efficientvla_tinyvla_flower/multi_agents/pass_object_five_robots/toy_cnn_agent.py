from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from train.toy_cnn.multi_agents.place_cucumber.model import HeteroMAPPOAgent
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.model import (
    build_batch_from_obs,
)


class PassObjectToyCNNAgent(HeteroMAPPOAgent):
    def __init__(
        self,
        *,
        agent_names,
        state_dims,
        global_state_dim,
        action_dims,
        normalize_state: bool = True,
        **_: Any,
    ):
        super().__init__(
            agent_names=agent_names,
            state_dims=state_dims,
            global_state_dim=global_state_dim,
            action_dims=action_dims,
            normalize_state=normalize_state,
        )
        self.action_position_placeholders = nn.ModuleDict(
            {
                name: nn.ModuleList([nn.Identity() for _ in range(self.action_dims[name])])
                for name in self.agent_names
            }
        )
        self.action_position_actor_placeholders = nn.ModuleDict(
            {
                name: nn.ModuleList([nn.Identity() for _ in range(self.action_dims[name])])
                for name in self.agent_names
            }
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare_batch(self, obs_or_batch: Mapping[str, Any]) -> Dict[str, Any]:
        first_agent = self.agent_names[0]
        if "global_state" in obs_or_batch and f"agent_states_{first_agent}" in obs_or_batch:
            batch = dict(obs_or_batch)
        else:
            batch = build_batch_from_obs(obs_or_batch, self.agent_names)
        numeric_keys = {"global_state", *(f"agent_states_{name}" for name in self.agent_names)}
        for key in numeric_keys:
            value = batch[key]
            if torch.is_tensor(value):
                tensor = value.to(dtype=torch.float32)
            else:
                tensor = torch.as_tensor(value, dtype=torch.float32)
            batch[key] = torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)
        rgb = batch["rgb"]
        if torch.is_tensor(rgb):
            rgb = rgb.to(dtype=torch.float32)
        else:
            rgb = torch.as_tensor(rgb, dtype=torch.float32)
        rgb = torch.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
        if rgb.ndim == 4 and rgb.shape[-1] in (3, 4):
            rgb = rgb[..., :3].permute(0, 3, 1, 2).contiguous()
        elif rgb.ndim == 3 and rgb.shape[-1] in (3, 4):
            rgb = rgb[..., :3].permute(2, 0, 1).unsqueeze(0).contiguous()
        if rgb.numel() > 0:
            max_value = float(rgb.detach().amax().item())
            if max_value > 1.5:
                rgb = rgb / 255.0
        if rgb.shape[-2:] != (128, 128):
            rgb = F.interpolate(rgb, size=(128, 128), mode="bilinear", align_corners=False)
        batch["rgb"] = rgb
        return batch

    def _aggregate_actor_feature(self, name: str, rgb_feat: torch.Tensor, agent_state: torch.Tensor) -> torch.Tensor:
        actor_feat = self._actor_feature(name, rgb_feat, agent_state)
        position_features = []
        for feature_layer, actor_layer in zip(
            self.action_position_placeholders[name],
            self.action_position_actor_placeholders[name],
        ):
            position_feature = feature_layer(actor_feat)
            position_feature = actor_layer(position_feature)
            position_features.append(position_feature)
        if len(position_features) == 1:
            return position_features[0]
        return torch.stack(position_features, dim=1).mean(dim=1)

    def get_action_and_value(
        self,
        batch,
        actions_input=None,
        action_bins_input=None,
        deterministic: bool = False,
        return_token_logits: bool = False,
    ):
        batch = self._prepare_batch(batch)
        if action_bins_input is not None and actions_input is None:
            actions_input = action_bins_input
        if deterministic and actions_input is None:
            actions_input = self.get_action(batch, deterministic=True)
        rgb = batch["rgb"]
        global_state = self._normalize_global_state(batch["global_state"])
        rgb_feat = self.rgb_encoder(rgb)
        critic_feat = self._critic_feature(rgb_feat, global_state)
        value = self.critic(critic_feat).squeeze(-1)

        actions_out = {}
        log_probs = {}
        entropies = {}
        for name in self.agent_names:
            agent_state = self._normalize_actor_state(name, batch[f"agent_states_{name}"])
            actor_feat = self._aggregate_actor_feature(name, rgb_feat, agent_state)
            mean = self.actor_heads[name](actor_feat)
            logstd = self.actor_logstd[name].expand_as(mean)
            mean, std = self._safe_distribution(mean, logstd)
            dist = Normal(mean, std)

            if actions_input is None:
                action = mean if deterministic else dist.sample()
            else:
                action = self._safe_action(actions_input[name], mean_ref=mean)

            actions_out[name] = action.detach().cpu().numpy()
            log_probs[name] = dist.log_prob(action).sum(-1)
            entropies[name] = dist.entropy().sum(-1)

        if return_token_logits:
            return actions_out, log_probs, entropies, value, None
        return actions_out, log_probs, entropies, value

    @torch.no_grad()
    def get_action(self, batch, deterministic=False):
        batch = self._prepare_batch(batch)
        rgb = batch["rgb"]
        rgb_feat = self.rgb_encoder(rgb)
        actions = {}
        for name in self.agent_names:
            state = self._normalize_actor_state(name, batch[f"agent_states_{name}"])
            feat = self._aggregate_actor_feature(name, rgb_feat, state)
            mean = self.actor_heads[name](feat)
            mean, std = self._safe_distribution(mean, self.actor_logstd[name].expand_as(mean))
            actions[name] = mean if deterministic else Normal(mean, std).sample()
        return actions

    @torch.no_grad()
    def get_value(self, batch):
        batch = self._prepare_batch(batch)
        rgb = batch["rgb"]
        global_state = self._normalize_global_state(batch["global_state"])
        rgb_feat = self.rgb_encoder(rgb)
        critic_feat = self._critic_feature(rgb_feat, global_state)
        return self.critic(critic_feat).squeeze(-1)

    @torch.no_grad()
    def update_state_stats(
        self,
        obs: Mapping[str, Any],
        *,
        update_actor: bool = True,
        update_critic: bool = True,
    ) -> None:
        parsed = self._prepare_batch(obs)
        if update_actor and self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].update(
                    parsed[f"agent_states_{name}"].to(device=self.device, dtype=torch.float32)
                )
        if update_critic and self.critic_state_rms is not None:
            self.critic_state_rms.update(
                parsed["global_state"].to(device=self.device, dtype=torch.float32)
            )

    def freeze_state_stats(self) -> None:
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].freeze()
        if self.critic_state_rms is not None:
            self.critic_state_rms.freeze()

    def unfreeze_state_stats(self) -> None:
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].unfreeze()
        if self.critic_state_rms is not None:
            self.critic_state_rms.unfreeze()

    def checkpoint_state_dict(self):
        return self.state_dict()

    def load_checkpoint_state_dict(self, state_dict):
        target_state = self.state_dict()
        matched = {}
        skipped = []
        for key, value in state_dict.items():
            if key not in target_state or target_state[key].shape != value.shape:
                skipped.append(key)
                continue
            matched[key] = value
        if not matched:
            raise RuntimeError(f"No compatible parameters found for {self.__class__.__name__}")
        merged_state = dict(target_state)
        merged_state.update(matched)
        self.load_state_dict(merged_state, strict=True)
        missing = sorted(set(target_state.keys()) - set(matched.keys()))
        print(
            f"[Checkpoint] {self.__class__.__name__} "
            f"matched={len(matched)} skipped={len(skipped)} missing={len(missing)}"
        )


def _trainable(params):
    return [parameter for parameter in params if parameter.requires_grad]


def build_toy_cnn_mappo_optimizer(args, agent: PassObjectToyCNNAgent) -> torch.optim.Optimizer:
    param_groups = [
        {
            "params": _trainable(agent.rgb_encoder.parameters()),
            "lr": args.backbone_learning_rate,
            "group_name": "vla_adapter_backbone",
        },
        {
            "params": _trainable(agent.actor_state_encoders.parameters()),
            "lr": args.state_learning_rate,
            "group_name": "vla_adapter_state_projector",
        },
        {
            "params": _trainable(agent.actor_heads.parameters()) + _trainable(agent.actor_logstd.parameters()),
            "lr": args.head_learning_rate,
            "group_name": "vla_adapter_heads",
        },
        {
            "params": _trainable(agent.critic_state_encoder.parameters()),
            "lr": args.value_head_learning_rate,
            "group_name": "critic_state_encoder",
        },
        {
            "params": _trainable(agent.critic.parameters()),
            "lr": args.value_head_learning_rate,
            "group_name": "critic",
        },
    ]
    param_groups = [group for group in param_groups if group["params"]]
    return torch.optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)
