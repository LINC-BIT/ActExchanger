from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.reinforcement_learning.utils import RunningMeanStd
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.model import build_batch_from_obs
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.model import extract_planner_subtasks_from_batch
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.tiny_vla import (
    MLPProjector,
    SharedTinyVLA4DActor,
    TinyVLABackbone,
)


class TinyContinuousVLAActor(nn.Module):
    """Small continuous VLA actor used for SmolVLA/EfficientVLA-style branches."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        image_size: int = 112,
        hidden_dim: int = 384,
        vision_layers: int = 6,
        decoder_layers: int = 1,
        attention_heads: int = 6,
        patch_size: int = 14,
        ffn_mult: int = 4,
        prompt_length: int = 1,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.image_size = int(image_size)
        self.hidden_dim = int(hidden_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.vla = TinyVLABackbone(
            image_size=self.image_size,
            patch_size=int(patch_size),
            hidden_dim=self.hidden_dim,
            vision_layers=int(vision_layers),
            decoder_layers=int(decoder_layers),
            attention_heads=int(attention_heads),
            ffn_mult=int(ffn_mult),
            prompt_length=int(prompt_length),
            max_action_dim=self.action_dim,
            num_action_bins=2,
        )
        self.state_projector = MLPProjector(
            input_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        ).to(dtype=torch.float32)
        self.context_projector = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ).to(dtype=torch.float32)
        self.action_mean_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        ).to(dtype=torch.float32)
        self.log_std_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        ).to(dtype=torch.float32)
        self.max_planner_subtask_bytes = 64
        self.subtask_byte_embedding = nn.Embedding(257, self.hidden_dim)
        self.subtask_projector = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ).to(dtype=torch.float32)

    @property
    def device(self) -> torch.device:
        return next(self.vla.parameters()).device

    def configure_trainable_modules(self, train_backbone: bool) -> None:
        for parameter in self.vla.parameters():
            parameter.requires_grad = bool(train_backbone)
        for module in (
            self.state_projector,
            self.context_projector,
            self.action_mean_head,
            self.log_std_head,
            self.subtask_byte_embedding,
            self.subtask_projector,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _encode_planner_subtasks(
        self,
        planner_subtasks: Optional[Sequence[Optional[str]]],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if planner_subtasks is None:
            return None
        if len(planner_subtasks) != batch_size:
            raise ValueError(f"planner_subtasks expects {batch_size} items, got {len(planner_subtasks)}")

        byte_ids = torch.zeros(
            batch_size,
            self.max_planner_subtask_bytes,
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.zeros_like(byte_ids, dtype=torch.bool)
        has_nonempty = False
        for row, value in enumerate(planner_subtasks):
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            encoded = text.encode("utf-8", errors="ignore")[: self.max_planner_subtask_bytes]
            if not encoded:
                continue
            ids = torch.as_tensor([byte + 1 for byte in encoded], dtype=torch.long, device=self.device)
            byte_ids[row, : ids.shape[0]] = ids
            mask[row, : ids.shape[0]] = True
            has_nonempty = True

        if not has_nonempty:
            return None
        embedded = self.subtask_byte_embedding(byte_ids)
        masked_embedded = embedded * mask.unsqueeze(-1).to(dtype=embedded.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=embedded.dtype)
        pooled = masked_embedded.sum(dim=1) / denom
        return self.subtask_projector(pooled.to(torch.float32))

    def _prepare_pixel_values(self, rgbs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(rgbs, torch.Tensor):
            rgb_batch = rgbs[..., :3].detach().to(device=self.device, dtype=torch.float32)
        else:
            rgb_batch = torch.as_tensor(np.asarray(rgbs)[..., :3], device=self.device, dtype=torch.float32)
        if rgb_batch.ndim != 4:
            raise ValueError(f"Expected RGB batch [B, H, W, 3], got {rgb_batch.shape}")
        rgb_batch = rgb_batch.permute(0, 3, 1, 2).contiguous() / 255.0
        if rgb_batch.shape[-2:] != (self.image_size, self.image_size):
            rgb_batch = F.interpolate(
                rgb_batch,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return rgb_batch

    def encode_visual_feature(self, rgbs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        pixel_values = self._prepare_pixel_values(rgbs)
        return self.vla.encode_vision(pixel_values).mean(dim=1).to(torch.float32)

    def sample_actions(
        self,
        rgbs: Union[np.ndarray, torch.Tensor],
        states: torch.Tensor,
        *,
        state_features: Optional[torch.Tensor] = None,
        actions_input: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        action_position_placeholders: Optional[nn.ModuleList] = None,
        action_position_actor_placeholders: Optional[nn.ModuleList] = None,
        planner_subtasks: Optional[Sequence[Optional[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        vision_feature = self.encode_visual_feature(rgbs)
        planner_feature = self._encode_planner_subtasks(planner_subtasks, vision_feature.shape[0])
        if planner_feature is not None:
            vision_feature = vision_feature + planner_feature.to(dtype=vision_feature.dtype)
        if state_features is None:
            state_tensor = torch.as_tensor(states, device=self.device, dtype=torch.float32)
            state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
            state_feature = self.state_projector(state_tensor)
        else:
            state_feature = torch.as_tensor(state_features, device=self.device, dtype=torch.float32)
            state_feature = torch.nan_to_num(state_feature, nan=0.0, posinf=1e4, neginf=-1e4)
        context_feature = self.context_projector(torch.cat([vision_feature, state_feature], dim=-1))
        policy_feature = context_feature
        action_position_features = None
        if action_position_placeholders is not None:
            if len(action_position_placeholders) != self.action_dim:
                raise ValueError(
                    f"Expected {self.action_dim} action-position placeholders, got {len(action_position_placeholders)}"
                )
            if (
                action_position_actor_placeholders is not None
                and len(action_position_actor_placeholders) != self.action_dim
            ):
                raise ValueError("action_position_actor_placeholders must match action_dim")
            per_action_features = []
            for action_idx in range(self.action_dim):
                pos_feature = action_position_placeholders[action_idx](context_feature)
                if action_position_actor_placeholders is not None:
                    pos_feature = action_position_actor_placeholders[action_idx](pos_feature)
                per_action_features.append(pos_feature)
            action_position_features = torch.stack(per_action_features, dim=1)
            policy_feature = action_position_features.mean(dim=1)
        mean = torch.tanh(self.action_mean_head(policy_feature))
        log_std = torch.clamp(self.log_std_head(policy_feature), min=self.log_std_min, max=self.log_std_max)
        dist = torch.distributions.Normal(mean, log_std.exp())

        if actions_input is not None:
            actions = torch.as_tensor(actions_input, device=self.device, dtype=torch.float32)
        elif deterministic:
            actions = dist.mean
        else:
            actions = dist.rsample()
        actions = torch.clamp(actions, -1.0, 1.0)
        return {
            "actions": actions,
            "log_prob": dist.log_prob(actions).sum(dim=-1),
            "entropy": dist.entropy().sum(dim=-1),
            "mean": mean,
            "std": log_std.exp(),
            "context_feature": context_feature,
            "policy_feature": policy_feature,
            "action_position_features": action_position_features,
        }


class MixedFiveVLAAgent(nn.Module):
    """Five-branch mixed agent for PassObjectFiveRobots.

    Branch order:
    1. VLA-Adapter
    2. SmolVLA
    3. EfficientVLA
    4. TinyVLA
    5. FLOWER
    """

    BRANCH_ORDER = ("vla_adapter", "smolvla", "efficientvla", "tinyvla", "flower")

    def __init__(
        self,
        agent_names: List[str],
        state_dim: int,
        global_state_dim: int,
        action_dim: Optional[int] = None,
        action_dims: Optional[Dict[str, int]] = None,
        state_dims: Optional[Dict[str, int]] = None,
        model_dir=None,
        normalize_state: bool = True,
        freeze_vla_backbone: bool = False,
        critic_hidden_dim: int = 512,
        attention_implementation: str = "sdpa",
        image_size: Optional[int] = None,
        use_vla_lora: bool = False,
        use_vision_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        train_vision_backbone: bool = False,
        vision_token_pool_size: Optional[int] = None,
        policy_mode: str = "native",
        tiny_hidden_dim: int = 384,
        tiny_vision_layers: int = 4,
        tiny_decoder_layers: int = 4,
        tiny_attention_heads: int = 6,
        tiny_patch_size: int = 14,
        tiny_ffn_mult: int = 4,
        tiny_num_action_bins: int = 256,
        tiny_prompt_length: int = 24,
        tiny_vla_use_decode_cache: bool = False,
        smolvla_hidden_dim: int = 384,
        smolvla_vision_layers: int = 6,
        smolvla_attention_heads: int = 6,
        smolvla_patch_size: int = 14,
        smolvla_ffn_mult: int = 4,
        efficientvla_hidden_dim: int = 256,
        efficientvla_vision_layers: int = 4,
        efficientvla_decoder_layers: int = 1,
        efficientvla_attention_heads: int = 4,
        efficientvla_patch_size: int = 16,
        efficientvla_ffn_mult: int = 3,
        tinyvla_hidden_dim: int = 320,
        tinyvla_vision_layers: int = 4,
        tinyvla_decoder_layers: int = 2,
        tinyvla_attention_heads: int = 4,
        tinyvla_patch_size: int = 16,
        tinyvla_ffn_mult: int = 3,
        flower_hidden_dim: int = 320,
        flower_vision_layers: int = 4,
        flower_decoder_layers: int = 2,
        flower_attention_heads: int = 4,
        flower_patch_size: int = 16,
        flower_ffn_mult: int = 3,
    ):
        super().__init__()
        if len(agent_names) != 5:
            raise ValueError(
                f"MixedFiveVLAAgent expects exactly 5 agents, got {agent_names}"
            )
        self.agent_names = list(agent_names)
        if state_dims is None:
            self.state_dims = {name: int(state_dim) for name in self.agent_names}
        else:
            self.state_dims = {name: int(state_dims[name]) for name in self.agent_names}
        if action_dims is None:
            if action_dim is None:
                raise ValueError("Provide action_dims for heterogeneous pass-object agents")
            self.action_dims = {name: int(action_dim) for name in self.agent_names}
        else:
            self.action_dims = {name: int(action_dims[name]) for name in self.agent_names}

        self.state_dim = int(state_dim)
        self.global_state_dim = int(global_state_dim)
        self.continuous_num_action_bins = int(tiny_num_action_bins)

        self.branch_name_to_agent = {
            branch_name: self.agent_names[idx]
            for idx, branch_name in enumerate(self.BRANCH_ORDER)
        }
        self.agent_to_branch_name = {
            agent_name: branch_name for branch_name, agent_name in self.branch_name_to_agent.items()
        }

        self.vla_agent_name = self.branch_name_to_agent["vla_adapter"]
        self.smolvla_agent_name = self.branch_name_to_agent["smolvla"]
        self.efficientvla_agent_name = self.branch_name_to_agent["efficientvla"]
        self.tinyvla_agent_name = self.branch_name_to_agent["tinyvla"]
        self.flower_agent_name = self.branch_name_to_agent["flower"]

        resolved_image_size = 112 if image_size is None else int(image_size)
        if resolved_image_size % int(efficientvla_patch_size) != 0:
            efficientvla_patch_size = 14
        if resolved_image_size % int(tinyvla_patch_size) != 0:
            tinyvla_patch_size = 14
        if resolved_image_size % int(flower_patch_size) != 0:
            flower_patch_size = 14

        self.vla_actor = SharedTinyVLA4DActor(
            model_dir=model_dir,
            state_dim=self.state_dims[self.vla_agent_name],
            env_action_dim=self.action_dims[self.vla_agent_name],
            attention_implementation=attention_implementation,
            image_size=resolved_image_size,
            use_lora=use_vla_lora,
            use_vision_lora=use_vision_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            train_vision_backbone=train_vision_backbone,
            vision_token_pool_size=vision_token_pool_size,
            policy_mode=policy_mode,
            tiny_hidden_dim=tiny_hidden_dim,
            tiny_vision_layers=tiny_vision_layers,
            tiny_decoder_layers=tiny_decoder_layers,
            tiny_attention_heads=tiny_attention_heads,
            tiny_patch_size=tiny_patch_size,
            tiny_ffn_mult=tiny_ffn_mult,
            tiny_num_action_bins=tiny_num_action_bins,
            tiny_prompt_length=tiny_prompt_length,
            use_decode_cache=tiny_vla_use_decode_cache,
        )
        self.vla_actor.configure_trainable_modules(not freeze_vla_backbone)

        self.smolvla_actor = TinyContinuousVLAActor(
            state_dim=self.state_dims[self.smolvla_agent_name],
            action_dim=self.action_dims[self.smolvla_agent_name],
            image_size=resolved_image_size,
            hidden_dim=smolvla_hidden_dim,
            vision_layers=smolvla_vision_layers,
            decoder_layers=1,
            attention_heads=smolvla_attention_heads,
            patch_size=smolvla_patch_size,
            ffn_mult=smolvla_ffn_mult,
            prompt_length=1,
        )
        self.smolvla_actor.configure_trainable_modules(not freeze_vla_backbone)

        self.efficientvla_actor = TinyContinuousVLAActor(
            state_dim=self.state_dims[self.efficientvla_agent_name],
            action_dim=self.action_dims[self.efficientvla_agent_name],
            image_size=resolved_image_size,
            hidden_dim=efficientvla_hidden_dim,
            vision_layers=efficientvla_vision_layers,
            decoder_layers=efficientvla_decoder_layers,
            attention_heads=efficientvla_attention_heads,
            patch_size=efficientvla_patch_size,
            ffn_mult=efficientvla_ffn_mult,
            prompt_length=1,
        )
        self.efficientvla_actor.configure_trainable_modules(not freeze_vla_backbone)

        self.tinyvla_actor = TinyContinuousVLAActor(
            state_dim=self.state_dims[self.tinyvla_agent_name],
            action_dim=self.action_dims[self.tinyvla_agent_name],
            image_size=resolved_image_size,
            hidden_dim=tinyvla_hidden_dim,
            vision_layers=tinyvla_vision_layers,
            decoder_layers=tinyvla_decoder_layers,
            attention_heads=tinyvla_attention_heads,
            patch_size=tinyvla_patch_size,
            ffn_mult=tinyvla_ffn_mult,
            prompt_length=1,
        )
        self.tinyvla_actor.configure_trainable_modules(not freeze_vla_backbone)

        self.flower_actor = TinyContinuousVLAActor(
            state_dim=self.state_dims[self.flower_agent_name],
            action_dim=self.action_dims[self.flower_agent_name],
            image_size=resolved_image_size,
            hidden_dim=flower_hidden_dim,
            vision_layers=flower_vision_layers,
            decoder_layers=flower_decoder_layers,
            attention_heads=flower_attention_heads,
            patch_size=flower_patch_size,
            ffn_mult=flower_ffn_mult,
            prompt_length=1,
        )
        self.flower_actor.configure_trainable_modules(not freeze_vla_backbone)

        self.branch_hidden_dims = {
            self.vla_agent_name: int(self.vla_actor.hidden_dim),
            self.smolvla_agent_name: int(self.smolvla_actor.hidden_dim),
            self.efficientvla_agent_name: int(self.efficientvla_actor.hidden_dim),
            self.tinyvla_agent_name: int(self.tinyvla_actor.hidden_dim),
            self.flower_agent_name: int(self.flower_actor.hidden_dim),
        }

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

        if normalize_state:
            self.actor_state_rms = nn.ModuleDict(
                {name: RunningMeanStd(shape=(self.state_dims[name],)) for name in self.agent_names}
            )
            self.critic_state_rms = RunningMeanStd(shape=(self.global_state_dim,))
        else:
            self.actor_state_rms = None
            self.critic_state_rms = None

        critic_hidden = int(max(self.branch_hidden_dims.values()))
        critic_visual_input_dim = int(sum(self.branch_hidden_dims.values()))
        self.critic_state_encoder = nn.Sequential(
            nn.LayerNorm(self.global_state_dim),
            nn.Linear(self.global_state_dim, critic_hidden),
            nn.GELU(),
            nn.Linear(critic_hidden, critic_hidden),
        ).to(dtype=torch.float32)
        self.critic_visual_encoder = nn.Sequential(
            nn.LayerNorm(critic_visual_input_dim),
            nn.Linear(critic_visual_input_dim, critic_hidden),
            nn.GELU(),
            nn.Linear(critic_hidden, critic_hidden),
        ).to(dtype=torch.float32)
        self.critic = nn.Sequential(
            nn.LayerNorm(critic_hidden * 2),
            nn.Linear(critic_hidden * 2, int(critic_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(critic_hidden_dim), 1),
        ).to(dtype=torch.float32)

    @property
    def device(self) -> torch.device:
        return self.vla_actor.device

    @property
    def actor(self):
        return self.vla_actor

    @property
    def vla_actor_action_position_placeholders(self):
        return self.action_position_placeholders[self.vla_agent_name]

    @property
    def vla_actor_action_position_actor_placeholders(self):
        return self.action_position_actor_placeholders[self.vla_agent_name]

    @property
    def smolvla_actor_action_position_placeholders(self):
        return self.action_position_placeholders[self.smolvla_agent_name]

    @property
    def smolvla_actor_action_position_actor_placeholders(self):
        return self.action_position_actor_placeholders[self.smolvla_agent_name]

    @property
    def efficientvla_actor_action_position_placeholders(self):
        return self.action_position_placeholders[self.efficientvla_agent_name]

    @property
    def efficientvla_actor_action_position_actor_placeholders(self):
        return self.action_position_actor_placeholders[self.efficientvla_agent_name]

    @property
    def tinyvla_actor_action_position_placeholders(self):
        return self.action_position_placeholders[self.tinyvla_agent_name]

    @property
    def tinyvla_actor_action_position_actor_placeholders(self):
        return self.action_position_actor_placeholders[self.tinyvla_agent_name]

    @property
    def flower_actor_action_position_placeholders(self):
        return self.action_position_placeholders[self.flower_agent_name]

    @property
    def flower_actor_action_position_actor_placeholders(self):
        return self.action_position_actor_placeholders[self.flower_agent_name]

    def requires_action_bins(self) -> Dict[str, bool]:
        return {name: False for name in self.agent_names}

    def _branch_actor(self, agent_name: str) -> nn.Module:
        branch_name = self.agent_to_branch_name[agent_name]
        if branch_name == "vla_adapter":
            return self.vla_actor
        return getattr(self, f"{branch_name}_actor")

    def _normalize_state(self, state: torch.Tensor, agent_name: str) -> torch.Tensor:
        if self.actor_state_rms is None:
            return state
        return self.actor_state_rms[agent_name](state)

    def _normalize_global_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.critic_state_rms is None:
            return state
        return self.critic_state_rms(state)

    def _agent_state(self, batch: Dict[str, Any], name: str) -> torch.Tensor:
        state = batch[f"agent_states_{name}"].to(device=self.device, dtype=torch.float32)
        return self._normalize_state(state, name)

    def _state_projector(self, agent_name: str) -> nn.Module:
        return self._branch_actor(agent_name).state_projector

    def _continuous_actions_input(self, actions_input, name: str):
        if actions_input is None or name not in actions_input:
            return None
        return torch.as_tensor(actions_input[name], device=self.device, dtype=torch.float32)

    def _vla_action_bins_input(self, action_bins) -> torch.Tensor:
        if torch.is_tensor(action_bins):
            is_float = torch.is_floating_point(action_bins)
        else:
            is_float = np.asarray(action_bins).dtype.kind == "f"
        if is_float:
            action_tensor = torch.as_tensor(action_bins, device=self.device, dtype=torch.float32)
            return self.vla_actor.env_actions_to_bin_indices(action_tensor).to(device=self.device, dtype=torch.long)
        bins = torch.as_tensor(action_bins, device=self.device, dtype=torch.long)
        num_bins = int(getattr(self.vla_actor, "num_action_bins", 256))
        return torch.clamp(bins, min=0, max=num_bins - 1)

    def _continuous_action_bins_input(self, action_bins) -> torch.Tensor:
        if torch.is_tensor(action_bins):
            is_float = torch.is_floating_point(action_bins)
        else:
            is_float = np.asarray(action_bins).dtype.kind == "f"
        if is_float:
            return self._continuous_actions_to_bin_indices(action_bins)
        bins = torch.as_tensor(action_bins, device=self.device, dtype=torch.long)
        return torch.clamp(bins, min=0, max=self.continuous_num_action_bins - 1)

    def _continuous_actions_to_bin_indices(self, actions: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        action_tensor = torch.as_tensor(actions, device=self.device, dtype=torch.float32).clamp(-1.0, 1.0)
        scaled = (action_tensor + 1.0) * 0.5
        bins = torch.round(scaled * float(self.continuous_num_action_bins - 1))
        return bins.to(dtype=torch.long)

    def _continuous_bin_indices_to_actions(self, bins: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        bin_tensor = torch.as_tensor(bins, device=self.device, dtype=torch.float32)
        if self.continuous_num_action_bins <= 1:
            return torch.zeros_like(bin_tensor, dtype=torch.float32)
        scaled = bin_tensor / float(self.continuous_num_action_bins - 1)
        return (scaled * 2.0 - 1.0).clamp(-1.0, 1.0)

    def _continuous_actions_input_from_any(self, actions_input, action_bins_input, name: str):
        if action_bins_input is not None and name in action_bins_input:
            if torch.is_tensor(action_bins_input[name]):
                is_float = torch.is_floating_point(action_bins_input[name])
            else:
                is_float = np.asarray(action_bins_input[name]).dtype.kind == "f"
            if is_float:
                return torch.as_tensor(action_bins_input[name], device=self.device, dtype=torch.float32)
            return self._continuous_bin_indices_to_actions(action_bins_input[name])
        return self._continuous_actions_input(actions_input, name)

    def _adapt_actor_state_features(
        self,
        state_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return state_features

    def _compute_value(self, batch: Dict[str, Any]) -> torch.Tensor:
        global_state = self._normalize_global_state(
            batch["global_state"].to(device=self.device, dtype=torch.float32)
        )
        global_feature = self.critic_state_encoder(global_state)
        rgb = batch["rgb"]
        visual_features = []
        with torch.no_grad():
            visual_features.append(
                self.vla_actor._project_vision_features(
                    self.vla_actor._prepare_pixel_values(rgb)
                ).mean(dim=1).to(torch.float32)
            )
            for name in (
                self.smolvla_agent_name,
                self.efficientvla_agent_name,
                self.tinyvla_agent_name,
                self.flower_agent_name,
            ):
                visual_features.append(self._branch_actor(name).encode_visual_feature(rgb))
        visual_feature = self.critic_visual_encoder(torch.cat(visual_features, dim=-1))
        return self.critic(torch.cat([global_feature, visual_feature], dim=-1)).squeeze(-1)

    def _vla_out(
        self,
        rgb,
        state: torch.Tensor,
        *,
        action_bins_input=None,
        actions_input=None,
        deterministic: bool,
        state_features: Optional[torch.Tensor] = None,
        planner_subtasks: Optional[Sequence[Optional[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        action_bins = None
        if action_bins_input is not None and self.vla_agent_name in action_bins_input:
            action_bins = self._vla_action_bins_input(action_bins_input[self.vla_agent_name])
        elif actions_input is not None and self.vla_agent_name in actions_input:
            action_tensor = torch.as_tensor(
                actions_input[self.vla_agent_name],
                device=self.device,
                dtype=torch.float32,
            )
            action_bins = self.vla_actor.env_actions_to_bin_indices(action_tensor)
        return self.vla_actor.get_action_and_stats(
            rgbs=rgb,
            states=state,
            state_features=state_features,
            action_bins=action_bins,
            prompt_role_ids=torch.zeros(state.shape[0], device=self.device, dtype=torch.long),
            planner_subtasks=planner_subtasks,
            deterministic=deterministic,
        )

    def _extract_planner_subtasks(
        self,
        batch: Mapping[str, Any],
        batch_size: int,
    ) -> Optional[Dict[str, List[Optional[str]]]]:
        return extract_planner_subtasks_from_batch(batch, self.agent_names, batch_size)

    def _continuous_actor_out(
        self,
        agent_name: str,
        rgb,
        state: torch.Tensor,
        *,
        state_feature: torch.Tensor,
        actions_input=None,
        action_bins_input=None,
        deterministic: bool,
        planner_subtasks: Optional[Sequence[Optional[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        actor = self._branch_actor(agent_name)
        return actor.sample_actions(
            rgb,
            state,
            state_features=state_feature,
            actions_input=self._continuous_actions_input_from_any(actions_input, action_bins_input, agent_name),
            deterministic=deterministic,
            action_position_placeholders=self.action_position_placeholders[agent_name],
            action_position_actor_placeholders=self.action_position_actor_placeholders[agent_name],
            planner_subtasks=planner_subtasks,
        )

    def get_action_and_value(
        self,
        batch: Dict[str, Any],
        actions_input=None,
        action_bins_input=None,
        return_token_logits: bool = False,
        **kwargs,
    ):
        deterministic = bool(kwargs.get("deterministic", False))
        return_action_bins = bool(kwargs.get("return_action_bins", False))
        rgb = batch["rgb"]

        states = {name: self._agent_state(batch, name) for name in self.agent_names}
        planner_subtasks = self._extract_planner_subtasks(batch, next(iter(states.values())).shape[0])
        state_features = {
            name: self._state_projector(name)(states[name])
            for name in self.agent_names
        }
        state_features = self._adapt_actor_state_features(state_features)

        vla_planner_subtasks = None if planner_subtasks is None else planner_subtasks[self.vla_agent_name]
        vla_out = self._vla_out(
            rgb,
            states[self.vla_agent_name],
            action_bins_input=action_bins_input,
            actions_input=actions_input,
            deterministic=deterministic,
            state_features=state_features[self.vla_agent_name],
            planner_subtasks=vla_planner_subtasks,
        )
        if (
            vla_out.get("action_position_prompt_hidden") is not None
            and vla_out.get("action_position_context_feature") is not None
            and vla_out.get("state_feature") is not None
        ):
            vla_position_features = []
            for action_idx in range(vla_out["action_position_prompt_hidden"].shape[1]):
                position_feature = torch.cat(
                    [
                        vla_out["action_position_prompt_hidden"][:, action_idx, :],
                        vla_out["action_position_context_feature"][:, action_idx, :],
                        vla_out["state_feature"],
                    ],
                    dim=-1,
                )
                position_feature = self.action_position_placeholders[self.vla_agent_name][action_idx](position_feature)
                position_feature = self.action_position_actor_placeholders[self.vla_agent_name][action_idx](
                    position_feature
                )
                vla_position_features.append(position_feature)
            vla_out["action_position_features"] = torch.stack(vla_position_features, dim=1)

        branch_outputs: Dict[str, Dict[str, torch.Tensor]] = {self.vla_agent_name: vla_out}
        for name in (
            self.smolvla_agent_name,
            self.efficientvla_agent_name,
            self.tinyvla_agent_name,
            self.flower_agent_name,
        ):
            branch_outputs[name] = self._continuous_actor_out(
                name,
                rgb,
                states[name],
                state_feature=state_features[name],
                actions_input=actions_input,
                action_bins_input=action_bins_input,
                deterministic=deterministic,
                planner_subtasks=None if planner_subtasks is None else planner_subtasks[name],
            )

        actions_out = {
            self.vla_agent_name: vla_out["env_actions"].detach().to(dtype=torch.float32).cpu().numpy(),
        }
        log_probs = {self.vla_agent_name: vla_out["log_prob"]}
        entropies = {self.vla_agent_name: vla_out["entropy"]}
        for name in (
            self.smolvla_agent_name,
            self.efficientvla_agent_name,
            self.tinyvla_agent_name,
            self.flower_agent_name,
        ):
            actions_out[name] = branch_outputs[name]["actions"].detach().to(dtype=torch.float32).cpu().numpy()
            log_probs[name] = branch_outputs[name]["log_prob"]
            entropies[name] = branch_outputs[name]["entropy"]
        value = self._compute_value(batch)

        if return_action_bins:
            action_bins_out = {}
            if action_bins_input is not None:
                for name in self.agent_names:
                    if name not in action_bins_input:
                        continue
                    if name == self.vla_agent_name:
                        action_bins_out[name] = self._vla_action_bins_input(action_bins_input[name])
                    else:
                        action_bins_out[name] = self._continuous_action_bins_input(action_bins_input[name])
            for name in self.agent_names:
                if name in action_bins_out:
                    continue
                if name == self.vla_agent_name:
                    source = actions_input[name] if actions_input is not None and name in actions_input else actions_out[name]
                    action_bins_out[name] = self.vla_actor.env_actions_to_bin_indices(source).to(
                        device=self.device,
                        dtype=torch.long,
                    )
                else:
                    action_bins_out[name] = self._continuous_actions_to_bin_indices(actions_out[name])
            return actions_out, log_probs, entropies, value, action_bins_out

        if return_token_logits:
            return actions_out, log_probs, entropies, value, {self.vla_agent_name: vla_out["token_logits"]}
        return actions_out, log_probs, entropies, value

    @torch.no_grad()
    def get_action(self, batch: Dict[str, Any], deterministic: bool = False):
        actions_out, _, _, _ = self.get_action_and_value(batch, deterministic=deterministic)
        return actions_out

    def get_value(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self._compute_value(batch)

    def forward(self, batch: Dict[str, Any]):
        _, log_probs, _, values = self.get_action_and_value(batch)
        return log_probs, values

    @torch.no_grad()
    def update_state_stats(
        self,
        obs: Dict[str, Any],
        *,
        update_actor: bool = True,
        update_critic: bool = True,
    ) -> None:
        parsed = build_batch_from_obs(obs, self.agent_names)
        if update_actor and self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].update(
                    parsed[f"agent_states_{name}"].to(device=self.device, dtype=torch.float32)
                )
        if update_critic and self.critic_state_rms is not None:
            self.critic_state_rms.update(parsed["global_state"].to(device=self.device, dtype=torch.float32))

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
            raise RuntimeError("No compatible parameters found for mixed five VLA agent")
        merged_state = dict(target_state)
        merged_state.update(matched)
        self.load_state_dict(merged_state, strict=True)
        missing = sorted(set(target_state.keys()) - set(matched.keys()))
        print(
            "[Checkpoint] MixedFiveVLAAgent "
            f"matched={len(matched)} skipped={len(skipped)} missing={len(missing)}"
        )


def build_mixed_mappo_optimizer(
    args,
    agent: MixedFiveVLAAgent,
) -> torch.optim.Optimizer:
    param_groups = []

    def _append_group(params, lr, group_name):
        params = [p for p in params if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr, "group_name": group_name})

    _append_group(agent.vla_actor.vla.parameters(), args.backbone_learning_rate, "vla_adapter_backbone")
    _append_group(agent.vla_actor.state_projector.parameters(), args.state_learning_rate, "vla_adapter_state_projector")
    _append_group(
        list(agent.vla_actor.context_projector.parameters()) + list(agent.vla_actor.actor_head.parameters()),
        args.head_learning_rate,
        "vla_adapter_heads",
    )

    for branch_name in ("smolvla", "efficientvla", "tinyvla", "flower"):
        actor = getattr(agent, f"{branch_name}_actor")
        _append_group(actor.vla.parameters(), args.backbone_learning_rate, f"{branch_name}_backbone")
        _append_group(actor.state_projector.parameters(), args.state_learning_rate, f"{branch_name}_state_projector")
        _append_group(
            list(actor.context_projector.parameters())
            + list(actor.action_mean_head.parameters())
            + list(actor.log_std_head.parameters())
            + list(actor.subtask_projector.parameters())
            + list(actor.subtask_byte_embedding.parameters()),
            args.head_learning_rate,
            f"{branch_name}_heads",
        )

    _append_group(agent.critic_state_encoder.parameters(), args.value_head_learning_rate, "critic_state_encoder")
    _append_group(agent.critic_visual_encoder.parameters(), args.value_head_learning_rate, "critic_visual_encoder")
    _append_group(agent.critic.parameters(), args.value_head_learning_rate, "critic")
    return torch.optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)


MixedFiveVLA100MAgent = MixedFiveVLAAgent
MixedTinyVLAAdapterSmolVLAEfficientVLAAgent = MixedFiveVLAAgent
MixedAgent = MixedFiveVLAAgent
