from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from train.internVL.model import SharedInternVLActor, build_batch_from_obs
from train.reinforcement_learning.utils import RunningMeanStd


class MultiAgentLatentCoordinator(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2, num_heads: int = 8) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)


class FutureStateHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, future_state_dim: int, horizon: int = 1) -> None:
        super().__init__()
        self.future_state_dim = int(future_state_dim)
        self.horizon = int(horizon)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.horizon * self.future_state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.horizon == 1:
            return out.view(x.shape[0], self.future_state_dim)
        return out.view(x.shape[0], self.horizon, self.future_state_dim)


class MapleEdgeVLAAgent(nn.Module):
    """Multi-agent MAPLE-style agent for collaborative VLA control.

    This implementation keeps the repo's shared VLA backbone and unified action head, while
    adding an explicit latent coordination stage before per-agent action prediction.
    """

    def __init__(
        self,
        agent_names: List[str],
        state_dim: int,
        global_state_dim: int,
        action_dim: int,
        model_dir: Union[str, Path],
        normalize_state: bool = True,
        freeze_vla_backbone: bool = False,
        critic_hidden_dim: int = 512,
        attention_implementation: str = "sdpa",
        image_size: Optional[int] = 112,
        use_vla_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        latent_layers: int = 2,
        future_state_dim: Optional[int] = None,
        future_horizon: int = 1,
    ):
        super().__init__()
        self.agent_names = list(agent_names)
        self.num_agents = len(self.agent_names)
        self.state_dim = int(state_dim)
        self.global_state_dim = int(global_state_dim)
        self.action_dim = int(action_dim)

        self.actor = SharedInternVLActor(
            model_dir=Path(model_dir),
            state_dim=self.state_dim,
            env_action_dim=self.action_dim,
            attention_implementation=attention_implementation,
            image_size=image_size,
            use_lora=use_vla_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.actor.configure_trainable_modules(not freeze_vla_backbone)
        self.actor_heads = nn.ModuleDict({name: nn.Identity() for name in self.agent_names})
        self.actor_feature_placeholders = nn.ModuleDict({name: nn.Identity() for name in self.agent_names})

        if normalize_state:
            self.actor_state_rms = nn.ModuleDict(
                {name: RunningMeanStd(shape=(self.state_dim,)) for name in self.agent_names}
            )
            self.critic_state_rms = RunningMeanStd(shape=(self.global_state_dim,))
        else:
            self.actor_state_rms = None
            self.critic_state_rms = None

        hidden_dim = self.actor.hidden_dim
        fused_dim = hidden_dim * 3
        self.latent_in = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
        ).to(dtype=torch.float32)
        self.coordinator = MultiAgentLatentCoordinator(hidden_dim=hidden_dim, num_layers=latent_layers).to(
            dtype=torch.float32
        )
        self.prompt_refine = nn.Linear(hidden_dim, hidden_dim).to(dtype=torch.float32)
        self.context_refine = nn.Linear(hidden_dim, hidden_dim).to(dtype=torch.float32)
        self.state_refine = nn.Linear(hidden_dim, hidden_dim).to(dtype=torch.float32)

        self.critic_state_encoder = nn.Sequential(
            nn.LayerNorm(self.global_state_dim),
            nn.Linear(self.global_state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(dtype=torch.float32)
        critic_input_dim = hidden_dim * (1 + 3 * self.num_agents)
        self.critic = nn.Sequential(
            nn.LayerNorm(critic_input_dim),
            nn.Linear(critic_input_dim, critic_hidden_dim),
            nn.GELU(),
            nn.Linear(critic_hidden_dim, 1),
        ).to(dtype=torch.float32)

        if future_state_dim is None:
            future_state_dim = self.global_state_dim
        self.future_state_dim = int(future_state_dim)
        self.future_horizon = int(future_horizon)
        self.future_state_head = FutureStateHead(
            input_dim=hidden_dim * (1 + self.num_agents),
            hidden_dim=max(256, hidden_dim),
            future_state_dim=self.future_state_dim,
            horizon=self.future_horizon,
        ).to(dtype=torch.float32)

    @property
    def device(self) -> torch.device:
        return self.actor.device

    def configure_trainable_modules(self, freeze_vla_backbone: bool) -> None:
        self.actor.configure_trainable_modules(not freeze_vla_backbone)
        modules = [
            self.latent_in,
            self.coordinator,
            self.prompt_refine,
            self.context_refine,
            self.state_refine,
            self.critic_state_encoder,
            self.critic,
            self.future_state_head,
        ]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _normalize_actor_state(self, state: torch.Tensor, agent_name: str) -> torch.Tensor:
        if self.actor_state_rms is None:
            return state
        return self.actor_state_rms[agent_name](state)

    def _normalize_global_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.critic_state_rms is None:
            return state
        return self.critic_state_rms(state)

    def _repeat_rgb_for_agents(self, rgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(rgb, torch.Tensor):
            return torch.cat([rgb] * self.num_agents, dim=0)
        return np.concatenate([rgb] * self.num_agents, axis=0)

    def _prepare_agent_inputs(
        self,
        batch: Mapping[str, Any],
    ) -> Tuple[Union[np.ndarray, torch.Tensor], List[torch.Tensor], torch.Tensor]:
        rgb = batch["rgb"]
        repeated_rgb = self._repeat_rgb_for_agents(rgb)
        normalized_states = []
        prompt_role_ids = []
        for name in self.agent_names:
            state = batch[f"agent_states_{name}"].to(device=self.device, dtype=torch.float32)
            normalized_states.append(self._normalize_actor_state(state, name))
            role_id = 0 if name.endswith("-0") else 1
            prompt_role_ids.append(torch.full((state.shape[0],), role_id, dtype=torch.long, device=self.device))
        return repeated_rgb, normalized_states, torch.cat(prompt_role_ids, dim=0)

    def _resolve_action_bins(
        self,
        actions_input: Optional[Mapping[str, Union[np.ndarray, torch.Tensor]]],
        action_bins_input: Optional[Mapping[str, Union[np.ndarray, torch.Tensor]]],
    ) -> Optional[torch.Tensor]:
        if action_bins_input is not None:
            return torch.cat(
                [
                    torch.as_tensor(action_bins_input[name], device=self.device, dtype=torch.long)
                    for name in self.agent_names
                ],
                dim=0,
            )
        if actions_input is not None:
            return torch.cat(
                [self.actor.env_actions_to_bin_indices(actions_input[name]) for name in self.agent_names],
                dim=0,
            ).to(self.device, dtype=torch.long)
        return None

    def _run_maple_actor(
        self,
        batch: Mapping[str, Any],
        actions_input: Optional[Mapping[str, Union[np.ndarray, torch.Tensor]]] = None,
        action_bins_input: Optional[Mapping[str, Union[np.ndarray, torch.Tensor]]] = None,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        repeated_rgb, normalized_states, prompt_role_ids = self._prepare_agent_inputs(batch)
        cat_states = torch.cat(normalized_states, dim=0)
        action_bins = self._resolve_action_bins(actions_input=actions_input, action_bins_input=action_bins_input)

        base_logits, prompt_hidden, context_feature, state_feature = self.actor._compute_policy_features(
            rgbs=repeated_rgb,
            states=cat_states,
            prompt_role_ids=prompt_role_ids,
        )
        batch_size = normalized_states[0].shape[0]
        fused = torch.cat([prompt_hidden, context_feature, state_feature], dim=-1)
        latent_tokens = self.latent_in(fused).view(self.num_agents, batch_size, -1).permute(1, 0, 2).contiguous()
        coordinated = self.coordinator(latent_tokens)

        refined_prompt = []
        refined_context = []
        refined_state = []
        logits_list = []
        env_actions = {}
        log_probs = {}
        entropies = {}
        action_bins_out = {}
        critic_features = []

        for idx, name in enumerate(self.agent_names):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            token = coordinated[:, idx, :]
            prompt_i = prompt_hidden[start:end] + self.prompt_refine(token)
            context_i = context_feature[start:end] + self.context_refine(token)
            state_i = state_feature[start:end] + self.state_refine(token)
            action_features = prompt_i.unsqueeze(1) + self.actor.action_queries.unsqueeze(0)
            logits_i = self.actor.actor_head(action_features, state_i, context_i)
            logits_i = torch.nan_to_num(logits_i + base_logits[start:end], nan=0.0, posinf=20.0, neginf=-20.0)
            categorical = torch.distributions.Categorical(logits=logits_i)
            if action_bins is None:
                selected_bins = logits_i.argmax(dim=-1) if deterministic else categorical.sample()
            else:
                selected_bins = action_bins[start:end]
            env_actions[name] = self.actor.bin_indices_to_env_actions(selected_bins).detach().cpu().numpy()
            log_probs[name] = categorical.log_prob(selected_bins).sum(dim=-1)
            entropies[name] = categorical.entropy().mean(dim=-1)
            action_bins_out[name] = selected_bins
            refined_prompt.append(prompt_i)
            refined_context.append(context_i)
            refined_state.append(state_i)
            logits_list.append(logits_i)
            critic_features.append(torch.cat([prompt_i, context_i, state_i], dim=-1))

        global_state = batch["global_state"].to(device=self.device, dtype=torch.float32)
        global_state = self._normalize_global_state(global_state)
        global_feature = self.critic_state_encoder(global_state)
        value = self.critic(torch.cat([global_feature] + critic_features, dim=-1)).squeeze(-1)
        future_input = torch.cat([global_feature, coordinated.reshape(batch_size, -1)], dim=-1)
        aux_outputs = {
            "future_state": self.future_state_head(future_input),
            "coordinated_latent": coordinated,
            "joint_value": value,
        }
        return {
            "actions_out": env_actions,
            "log_probs": log_probs,
            "entropies": entropies,
            "action_bins_out": action_bins_out,
            "value": value,
            "aux_outputs": aux_outputs,
        }

    def get_action_and_value(
        self,
        batch: Mapping[str, Any],
        actions_input: Optional[Mapping[str, Union[np.ndarray, torch.Tensor]]] = None,
        action_bins_input: Optional[Mapping[str, Union[np.ndarray, torch.Tensor]]] = None,
        return_action_bins: bool = False,
        return_aux: bool = False,
        deterministic: bool = False,
    ):
        output = self._run_maple_actor(
            batch=batch,
            actions_input=actions_input,
            action_bins_input=action_bins_input,
            deterministic=deterministic,
        )
        if return_action_bins:
            return (
                output["actions_out"],
                output["log_probs"],
                output["entropies"],
                output["value"],
                output["action_bins_out"],
            )
        if return_aux:
            return (
                output["actions_out"],
                output["log_probs"],
                output["entropies"],
                output["value"],
                output["aux_outputs"],
            )
        return output["actions_out"], output["log_probs"], output["entropies"], output["value"]

    @torch.no_grad()
    def get_action(self, batch: Mapping[str, Any], deterministic: bool = False) -> Dict[str, np.ndarray]:
        actions_out, _, _, _ = self.get_action_and_value(batch, deterministic=deterministic)
        return actions_out

    def get_value(self, batch: Mapping[str, Any]) -> torch.Tensor:
        output = self._run_maple_actor(batch=batch, deterministic=True)
        return output["value"]

    @torch.no_grad()
    def update_state_stats(self, obs: Dict[str, Any]) -> None:
        parsed = build_batch_from_obs(obs, self.agent_names)
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].update(parsed[f"agent_states_{name}"].to(device=self.device, dtype=torch.float32))
        if self.critic_state_rms is not None:
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

    def checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        state = self.state_dict()
        if not self.actor.use_lora:
            return state
        filtered_state = {}
        for key, value in state.items():
            if key.startswith("actor.vla.") and "lora_" not in key.lower():
                continue
            filtered_state[key] = value
        return filtered_state

    def load_checkpoint_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if self.actor.use_lora:
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            unexpected = [key for key in unexpected if "lora_" not in key.lower()]
            missing = [key for key in missing if not (key.startswith("actor.vla.") and "lora_" not in key.lower())]
            if unexpected:
                raise RuntimeError(f"Unexpected keys when loading LoRA checkpoint: {unexpected}")
            if missing:
                raise RuntimeError(f"Missing keys when loading LoRA checkpoint: {missing}")
            return
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def predict_future_state(self, batch: Mapping[str, Any]) -> torch.Tensor:
        output = self._run_maple_actor(batch=batch, deterministic=True)
        return output["aux_outputs"]["future_state"]


def build_optimizer(args, agent: MapleEdgeVLAAgent) -> torch.optim.Optimizer:
    param_groups = [
        {
            "params": list(agent.actor.vla.parameters()),
            "lr": 0.0 if getattr(args, "freeze_vla_backbone", False) else args.backbone_learning_rate,
            "group_name": "vla",
        },
        {
            "params": list(agent.actor.state_projector.parameters()),
            "lr": args.state_learning_rate,
            "group_name": "state_projector",
        },
        {
            "params": list(agent.actor.context_projector.parameters()),
            "lr": args.head_learning_rate,
            "group_name": "context_projector",
        },
        {
            "params": list(agent.actor.actor_head.parameters()) + [agent.actor.action_queries],
            "lr": args.head_learning_rate,
            "group_name": "actor_head",
        },
        {
            "params": (
                list(agent.latent_in.parameters())
                + list(agent.coordinator.parameters())
                + list(agent.prompt_refine.parameters())
                + list(agent.context_refine.parameters())
                + list(agent.state_refine.parameters())
            ),
            "lr": args.head_learning_rate,
            "group_name": "latent_coordination",
        },
        {
            "params": list(agent.future_state_head.parameters()),
            "lr": getattr(args, "future_head_learning_rate", args.head_learning_rate),
            "group_name": "future_state_head",
        },
        {
            "params": list(agent.critic_state_encoder.parameters()) + list(agent.critic.parameters()),
            "lr": args.value_head_learning_rate,
            "group_name": "critic",
        },
    ]
    return torch.optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)
