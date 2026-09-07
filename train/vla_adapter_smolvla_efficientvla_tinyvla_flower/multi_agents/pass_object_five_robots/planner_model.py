from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from train.reinforcement_learning.utils import RunningMeanStd
from train.toy_cnn.model import make_mlp_with_orth_init
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.model import build_batch_from_obs


DEFAULT_SUBTASK_VOCAB = (
    "",
    "push the cube to the first handoff target",
    "move toward the cube and prepare contact",
    "push the cube to the second handoff target",
    "push the cube to the third handoff target",
    "grasp the cube and move above the handoff palm",
    "release the cube onto the receiving hand",
    "stabilize the cube inside the palm",
    "hold position and avoid blocking the next robot",
    "retract to a safe pose after handoff",
    "assist the next stage while avoiding collisions",
    "recover from a failed handoff and retry",
    "finish once the hand has secured the cube",
)

DEFAULT_PLANNER_STYLE_VOCAB = ("", "carefully", "steadily", "precisely", "cooperatively")
DEFAULT_PLANNER_COORDINATION_VOCAB = (
    "",
    "while coordinating with the other robots",
    "while keeping the handoff corridor clear",
    "while preparing for the next stage",
    "while avoiding collisions",
)


def _to_long_action_tensor(
    action: Any,
    device: torch.device,
    *,
    num_components: Optional[int] = None,
) -> torch.Tensor:
    if isinstance(action, torch.Tensor):
        tensor = action.to(device=device, dtype=torch.long)
    else:
        tensor = torch.as_tensor(action, device=device, dtype=torch.long)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    elif tensor.ndim != 2:
        raise ValueError(f"Expected planner action ids to be [B] or [B,K], got {tuple(tensor.shape)}")
    if num_components is not None and tensor.shape[-1] != int(num_components):
        raise ValueError(
            f"Expected planner action ids to have {num_components} components, got {tuple(tensor.shape)}"
        )
    return tensor


class HighLevelSubtaskPlannerAgent(nn.Module):
    STATE_ABS_CLAMP = 1e4
    LOGIT_ABS_CLAMP = 20.0

    def __init__(
        self,
        low_level_agent: nn.Module,
        agent_names: Sequence[str],
        state_dim: int,
        global_state_dim: int,
        action_dim: int,
        *,
        action_dims: Optional[Mapping[str, int]] = None,
        state_dims: Optional[Mapping[str, int]] = None,
        normalize_state: bool = True,
        planner_hidden_dim: int = 256,
        planner_layers: int = 2,
        planner_subtasks: Optional[Sequence[str]] = None,
        planner_text_mode: str = "fixed",
        planner_style_texts: Optional[Sequence[str]] = None,
        planner_coordination_texts: Optional[Sequence[str]] = None,
        low_level_deterministic: bool = True,
    ):
        super().__init__()
        self.low_level_agent = low_level_agent
        self.agent_names = list(agent_names)
        self.num_agents = len(self.agent_names)
        if state_dims is None and hasattr(low_level_agent, "state_dims"):
            state_dims = low_level_agent.state_dims
        self.state_dims = (
            {name: int(state_dims[name]) for name in self.agent_names}
            if state_dims is not None
            else {name: int(state_dim) for name in self.agent_names}
        )
        if action_dims is None and hasattr(low_level_agent, "action_dims"):
            action_dims = low_level_agent.action_dims
        self.state_dim = int(state_dim)
        self.global_state_dim = int(global_state_dim)
        self.action_dim = int(action_dim)
        self.action_dims = (
            {name: int(action_dims[name]) for name in self.agent_names}
            if action_dims is not None
            else {name: int(action_dim) for name in self.agent_names}
        )
        self.low_level_deterministic = bool(low_level_deterministic)
        self.subtask_texts = list(planner_subtasks or DEFAULT_SUBTASK_VOCAB)
        self.num_subtasks = len(self.subtask_texts)
        if self.num_subtasks < 2:
            raise ValueError("planner_subtasks must contain at least 2 entries")

        self.planner_text_mode = str(planner_text_mode).strip().lower()
        if self.planner_text_mode not in {"fixed", "tiny_text"}:
            raise ValueError(f"Unsupported planner_text_mode={planner_text_mode}")
        self.style_texts = list(planner_style_texts or DEFAULT_PLANNER_STYLE_VOCAB)
        self.coordination_texts = list(planner_coordination_texts or DEFAULT_PLANNER_COORDINATION_VOCAB)
        self.planner_action_components = ["subtask"]
        self.planner_component_vocab_sizes = {"subtask": self.num_subtasks}
        if self.planner_text_mode == "tiny_text":
            self.planner_action_components.extend(["style", "coordination"])
            self.planner_component_vocab_sizes["style"] = len(self.style_texts)
            self.planner_component_vocab_sizes["coordination"] = len(self.coordination_texts)
        self.num_planner_action_components = len(self.planner_action_components)

        for parameter in self.low_level_agent.parameters():
            parameter.requires_grad = False
        self.low_level_agent.eval()

        if normalize_state:
            self.actor_state_rms = nn.ModuleDict(
                {name: RunningMeanStd(shape=(self.state_dims[name],)) for name in self.agent_names}
            )
            self.critic_state_rms = RunningMeanStd(shape=(self.global_state_dim,))
        else:
            self.actor_state_rms = None
            self.critic_state_rms = None

        self.local_state_encoders = nn.ModuleDict(
            {
                name: make_mlp_with_orth_init(
                    self.state_dims[name],
                    [planner_hidden_dim, planner_hidden_dim],
                    last_act=True,
                )
                for name in self.agent_names
            }
        )
        self.global_state_encoder = make_mlp_with_orth_init(
            self.global_state_dim,
            [planner_hidden_dim, planner_hidden_dim],
            last_act=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=planner_hidden_dim,
            nhead=4,
            dim_feedforward=planner_hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.coordinator = nn.TransformerEncoder(layer, num_layers=max(1, int(planner_layers)))
        self.agent_id_embedding = nn.Parameter(torch.zeros(1, self.num_agents, planner_hidden_dim))
        nn.init.normal_(self.agent_id_embedding, mean=0.0, std=0.02)

        self.actor_heads = nn.ModuleDict()
        for name in self.agent_names:
            component_heads = nn.ModuleDict()
            for component_name in self.planner_action_components:
                component_heads[component_name] = make_mlp_with_orth_init(
                    planner_hidden_dim * 2,
                    [planner_hidden_dim, self.planner_component_vocab_sizes[component_name]],
                    last_act=False,
                    is_actor=True,
                )
            self.actor_heads[name] = component_heads
        self._initialize_actor_heads_near_null()
        self.critic = make_mlp_with_orth_init(
            planner_hidden_dim * (self.num_agents + 1),
            [planner_hidden_dim, 1],
            last_act=False,
        )

        self.reference_local_state_encoders = deepcopy(self.local_state_encoders)
        self.reference_global_state_encoder = deepcopy(self.global_state_encoder)
        self.reference_coordinator = deepcopy(self.coordinator)
        self.reference_actor_heads = deepcopy(self.actor_heads)
        self.reference_agent_id_embedding = nn.Parameter(
            self.agent_id_embedding.detach().clone(),
            requires_grad=False,
        )
        self._freeze_reference_modules()

    @property
    def device(self) -> torch.device:
        return next(self.global_state_encoder.parameters()).device

    @property
    def local_state_encoder(self) -> nn.Module:
        # Compatibility with the two_robot_pick planner optimizer helper.
        return self.local_state_encoders

    def _freeze_reference_modules(self) -> None:
        for module in (
            self.reference_local_state_encoders,
            self.reference_global_state_encoder,
            self.reference_coordinator,
            self.reference_actor_heads,
        ):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def _initialize_actor_heads_near_null(self) -> None:
        # Keep the planner almost neutral at initialization so the frozen
        # low-level policy stays close to its pretrained behavior until the
        # planner has learned useful subtask routing.
        for component_heads in self.actor_heads.values():
            for component_name, head in component_heads.items():
                linear_layers = [module for module in head.modules() if isinstance(module, nn.Linear)]
                if not linear_layers:
                    continue
                for module in linear_layers[:-1]:
                    nn.init.normal_(module.weight, mean=0.0, std=1e-3)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                final_linear = linear_layers[-1]
                if final_linear is None or final_linear.bias is None:
                    continue
                # Bias the initial deterministic planner output to the empty
                # option, yielding no extra planner text by default.
                nn.init.zeros_(final_linear.weight)
                nn.init.zeros_(final_linear.bias)
                final_linear.bias.data[0] = 1.0

    def sync_reference_policy(self) -> None:
        self.reference_local_state_encoders.load_state_dict(self.local_state_encoders.state_dict())
        self.reference_global_state_encoder.load_state_dict(self.global_state_encoder.state_dict())
        self.reference_coordinator.load_state_dict(self.coordinator.state_dict())
        self.reference_actor_heads.load_state_dict(self.actor_heads.state_dict())
        self.reference_agent_id_embedding.data.copy_(self.agent_id_embedding.detach())
        self._freeze_reference_modules()

    def _sanitize_tensor(self, tensor: torch.Tensor, *, clamp: float) -> torch.Tensor:
        tensor = torch.nan_to_num(
            tensor.float(),
            nan=0.0,
            posinf=clamp,
            neginf=-clamp,
        )
        return tensor.clamp(-clamp, clamp)

    def _normalize_actor_state(self, state: torch.Tensor, agent_name: str) -> torch.Tensor:
        state = self._sanitize_tensor(state, clamp=self.STATE_ABS_CLAMP)
        if self.actor_state_rms is None:
            return state
        return self._sanitize_tensor(self.actor_state_rms[agent_name](state), clamp=self.STATE_ABS_CLAMP)

    def _normalize_global_state(self, state: torch.Tensor) -> torch.Tensor:
        state = self._sanitize_tensor(state, clamp=self.STATE_ABS_CLAMP)
        if self.critic_state_rms is None:
            return state
        return self._sanitize_tensor(self.critic_state_rms(state), clamp=self.STATE_ABS_CLAMP)

    def _encode_policy_features(self, batch: Mapping[str, Any], *, use_reference: bool = False):
        local_hidden = []
        encoders = self.reference_local_state_encoders if use_reference else self.local_state_encoders
        for name in self.agent_names:
            state = batch[f"agent_states_{name}"].to(device=self.device, dtype=torch.float32)
            state = self._normalize_actor_state(state, name)
            local_hidden.append(self._sanitize_tensor(encoders[name](state), clamp=self.STATE_ABS_CLAMP))
        local_hidden = torch.stack(local_hidden, dim=1)
        local_hidden = self._sanitize_tensor(local_hidden, clamp=self.STATE_ABS_CLAMP)

        global_state = batch["global_state"].to(device=self.device, dtype=torch.float32)
        global_state = self._normalize_global_state(global_state)
        if use_reference:
            global_hidden = self._sanitize_tensor(
                self.reference_global_state_encoder(global_state),
                clamp=self.STATE_ABS_CLAMP,
            )
            coordinated = self._sanitize_tensor(
                self.reference_coordinator(
                    local_hidden + global_hidden.unsqueeze(1) + self.reference_agent_id_embedding
                ),
                clamp=self.STATE_ABS_CLAMP,
            )
        else:
            global_hidden = self._sanitize_tensor(
                self.global_state_encoder(global_state),
                clamp=self.STATE_ABS_CLAMP,
            )
            coordinated = self._sanitize_tensor(
                self.coordinator(local_hidden + global_hidden.unsqueeze(1) + self.agent_id_embedding),
                clamp=self.STATE_ABS_CLAMP,
            )

        actor_inputs = {
            name: self._sanitize_tensor(
                torch.cat([coordinated[:, idx, :], global_hidden], dim=-1),
                clamp=self.STATE_ABS_CLAMP,
            )
            for idx, name in enumerate(self.agent_names)
        }
        value_input = self._sanitize_tensor(
            torch.cat([global_hidden, coordinated.reshape(coordinated.shape[0], -1)], dim=-1),
            clamp=self.STATE_ABS_CLAMP,
        )
        value = self._sanitize_tensor(self.critic(value_input).squeeze(-1), clamp=self.STATE_ABS_CLAMP)
        return actor_inputs, value

    def _planner_distributions(self, batch: Mapping[str, Any], *, use_reference: bool = False):
        actor_inputs, value = self._encode_policy_features(batch, use_reference=use_reference)
        heads = self.reference_actor_heads if use_reference else self.actor_heads
        dists = {}
        for name in self.agent_names:
            dists[name] = {}
            for component_name in self.planner_action_components:
                logits = self._sanitize_tensor(
                    heads[name][component_name](actor_inputs[name]),
                    clamp=self.LOGIT_ABS_CLAMP,
                )
                dists[name][component_name] = Categorical(
                    logits=logits
                )
        return dists, value

    def _planner_actions_to_text(self, planner_action_ids: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        outputs: Dict[str, List[str]] = {}
        for name, ids in planner_action_ids.items():
            ids = ids.to(device="cpu", dtype=torch.long)
            subtask_ids = ids[:, 0].tolist()
            if self.planner_text_mode == "fixed":
                outputs[name] = [self.subtask_texts[int(idx)] for idx in subtask_ids]
                continue
            style_ids = ids[:, 1].tolist()
            coordination_ids = ids[:, 2].tolist()
            texts = []
            for subtask_idx, style_idx, coord_idx in zip(subtask_ids, style_ids, coordination_ids):
                subtask_text = self.subtask_texts[int(subtask_idx)].strip()
                style_text = self.style_texts[int(style_idx)].strip()
                coordination_text = self.coordination_texts[int(coord_idx)].strip()
                planner_text = subtask_text if not style_text else f"{style_text} {subtask_text}"
                if coordination_text:
                    planner_text = f"{planner_text} {coordination_text}"
                texts.append(planner_text.strip())
            outputs[name] = texts
        return outputs

    def _sample_planner_actions(self, dists, *, deterministic: bool, actions_input=None):
        planner_action_ids = {}
        log_probs = {}
        entropies = {}
        for name in self.agent_names:
            selected_components = []
            provided_actions = (
                None
                if actions_input is None
                else _to_long_action_tensor(
                    actions_input[name],
                    self.device,
                    num_components=self.num_planner_action_components,
                )
            )
            total_log_prob = None
            total_entropy = None
            for component_idx, component_name in enumerate(self.planner_action_components):
                dist = dists[name][component_name]
                if provided_actions is None:
                    selected = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
                else:
                    selected = provided_actions[:, component_idx]
                selected_components.append(selected)
                component_log_prob = dist.log_prob(selected)
                component_entropy = dist.entropy()
                total_log_prob = component_log_prob if total_log_prob is None else total_log_prob + component_log_prob
                total_entropy = component_entropy if total_entropy is None else total_entropy + component_entropy
            planner_action_ids[name] = torch.stack(selected_components, dim=-1)
            log_probs[name] = total_log_prob
            entropies[name] = total_entropy
        return planner_action_ids, log_probs, entropies

    def _placeholder_env_actions(self, batch_size: int) -> Dict[str, np.ndarray]:
        return {
            name: np.zeros((batch_size, self.action_dims[name]), dtype=np.float32)
            for name in self.agent_names
        }

    def rollout_step(self, batch: Mapping[str, Any], deterministic: bool = False):
        dists, value = self._planner_distributions(batch, use_reference=False)
        planner_action_ids, log_probs, entropies = self._sample_planner_actions(
            dists,
            deterministic=deterministic,
            actions_input=None,
        )
        low_level_batch = dict(batch)
        low_level_batch["planner_subtasks"] = self._planner_actions_to_text(planner_action_ids)
        env_actions = self.low_level_agent.get_action(
            low_level_batch,
            deterministic=self.low_level_deterministic,
        )
        return env_actions, planner_action_ids, log_probs, entropies, value

    def get_action_and_value(self, batch: Mapping[str, Any], actions_input=None):
        dists, value = self._planner_distributions(batch, use_reference=False)
        batch_size = batch["global_state"].shape[0]
        _, log_probs, entropies = self._sample_planner_actions(
            dists,
            deterministic=False,
            actions_input=actions_input,
        )
        return self._placeholder_env_actions(batch_size), log_probs, entropies, value

    @torch.no_grad()
    def get_reference_action_logprobs(self, batch: Mapping[str, Any], actions_input=None):
        if actions_input is None:
            raise ValueError("get_reference_action_logprobs requires actions_input")
        dists, _ = self._planner_distributions(batch, use_reference=True)
        outputs = {}
        for name in self.agent_names:
            selected = _to_long_action_tensor(
                actions_input[name],
                self.device,
                num_components=self.num_planner_action_components,
            )
            total_log_prob = None
            for component_idx, component_name in enumerate(self.planner_action_components):
                component_log_prob = dists[name][component_name].log_prob(selected[:, component_idx])
                total_log_prob = component_log_prob if total_log_prob is None else total_log_prob + component_log_prob
            outputs[name] = total_log_prob
        return outputs

    @torch.no_grad()
    def get_action(self, batch: Mapping[str, Any], deterministic: bool = False):
        env_actions, _, _, _, _ = self.rollout_step(batch, deterministic=deterministic)
        return env_actions

    def get_value(self, batch: Mapping[str, Any]) -> torch.Tensor:
        _, value = self._encode_policy_features(batch, use_reference=False)
        return value

    @torch.no_grad()
    def update_state_stats(self, obs_or_batch: Mapping[str, Any], *, update_actor: bool = True, update_critic: bool = True) -> None:
        batch = obs_or_batch
        if "global_state" not in batch:
            batch = build_batch_from_obs(batch, self.agent_names)
        if update_actor and self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].update(
                    batch[f"agent_states_{name}"].to(device=self.device, dtype=torch.float32)
                )
        if update_critic and self.critic_state_rms is not None:
            self.critic_state_rms.update(batch["global_state"].to(device=self.device, dtype=torch.float32))

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
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("low_level_agent.") and not key.startswith("reference_")
        }

    def load_checkpoint_state_dict(self, state_dict):
        target_state = self.state_dict()
        matched = {}
        for key, value in state_dict.items():
            if key in target_state and target_state[key].shape == value.shape:
                matched[key] = value
        if not matched:
            raise RuntimeError("No compatible parameters found for HighLevelSubtaskPlannerAgent")
        merged = dict(target_state)
        merged.update(matched)
        self.load_state_dict(merged, strict=False)
        self.sync_reference_policy()
