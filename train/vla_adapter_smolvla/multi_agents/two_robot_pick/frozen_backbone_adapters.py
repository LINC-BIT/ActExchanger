from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from train.marl.dicg.modules import DICGCore
from train.marl.mat.model import EncodeBlock, _init_linear
from train.marl.tgcnet.modules import TGCCommunicationBlock
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.mixed_sft_agent import (
    MixedTinyVLAAdapterSmolVLASFTAgent,
)
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.model import MultiAgentVLAAdapterMAPPOAgent


def _zero_last_linear(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
            return


def _split_agent_major_features(features: torch.Tensor, agent_names: Iterable[str]) -> torch.Tensor:
    names = list(agent_names)
    batch_size = features.shape[0] // max(len(names), 1)
    chunks = []
    for idx in range(len(names)):
        start = idx * batch_size
        end = (idx + 1) * batch_size
        chunks.append(features[start:end])
    return torch.stack(chunks, dim=1)


def _merge_agent_tokens(tokens: torch.Tensor) -> torch.Tensor:
    chunks = [tokens[:, idx, :] for idx in range(tokens.shape[1])]
    return torch.cat(chunks, dim=0)


class PerAgentProjectInOutAdapter(nn.Module):
    def __init__(self, input_dims: List[int], shared_dim: int, core: nn.Module):
        super().__init__()
        self.input_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, shared_dim),
                )
                for input_dim in input_dims
            ]
        )
        self.core = core
        self.output_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(shared_dim),
                    nn.Linear(shared_dim, input_dim),
                )
                for input_dim in input_dims
            ]
        )
        for proj in self.output_projs:
            _zero_last_linear(proj)

    def forward(self, feature_list: List[torch.Tensor]) -> List[torch.Tensor]:
        shared_tokens = torch.stack(
            [proj(feature) for proj, feature in zip(self.input_projs, feature_list)],
            dim=1,
        )
        residual_tokens = self.core(shared_tokens)
        return [
            proj(residual_tokens[:, idx, :])
            for idx, proj in enumerate(self.output_projs)
        ]


class FrozenBackboneAdapterAgent(MultiAgentVLAAdapterMAPPOAgent):
    def __init__(self, *args, **kwargs):
        model_backbone = str(kwargs.get("model_backbone", "openvla"))
        if model_backbone == "mixed_tiny_vla_smolvla":
            raise ValueError(
                "FrozenBackboneAdapterAgent currently supports only model_backbone=tiny or openvla. "
                "mixed_tiny_vla_smolvla needs a dedicated adapter path."
            )
        super().__init__(*args, **kwargs)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def _coordinator_residual(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _adapt_state_features(self, state_features: torch.Tensor) -> torch.Tensor:
        agent_tokens = _split_agent_major_features(state_features, self.agent_names)
        residual = self._coordinator_residual(agent_tokens)
        if residual.shape != agent_tokens.shape:
            raise RuntimeError(
                f"Coordinator residual shape mismatch: expected {tuple(agent_tokens.shape)}, got {tuple(residual.shape)}"
            )
        return _merge_agent_tokens(agent_tokens + residual)

    def _build_actor_input_dict(self, state_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return super()._build_actor_input_dict(self._adapt_state_features(state_features))

    @torch.no_grad()
    def update_state_stats(
        self,
        obs,
        *,
        update_actor: bool = True,
        update_critic: bool = True,
    ) -> None:
        del obs, update_actor, update_critic
        return None

    def freeze_state_stats(self) -> None:
        return None

    def unfreeze_state_stats(self) -> None:
        return None

    def checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.state_dict()

    def load_checkpoint_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        has_mixed_actor_prefix = any(key.startswith("vla_actor.") or key.startswith("smolvla_actor.") for key in state_dict)
        has_single_actor_prefix = any(key.startswith("actor.") for key in state_dict)
        if has_mixed_actor_prefix and not has_single_actor_prefix:
            raise RuntimeError(
                "Incompatible init checkpoint: detected mixed_tiny_vla_smolvla actor keys "
                "('vla_actor.' / 'smolvla_actor.') while this adapter wrapper is built on the "
                "single-backbone MultiAgentVLAAdapterMAPPOAgent ('actor.'). "
                "Use a matching tiny/openvla checkpoint under vla_adapter_smolvla/, or add a dedicated mixed-backbone adapter wrapper."
            )
        current_state = self.state_dict()
        matched = {}
        for key, value in state_dict.items():
            if key in current_state and current_state[key].shape == value.shape:
                matched[key] = value
        if not matched:
            raise RuntimeError(f"No compatible parameters found for {self.__class__.__name__}")
        merged_state = dict(current_state)
        merged_state.update(matched)
        self.load_state_dict(merged_state, strict=False)


class MATStateAdapter(nn.Module):
    def __init__(self, hidden_dim: int, num_agents: int, n_embd: int = 256, n_head: int = 4, n_block: int = 2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            _init_linear(nn.Linear(hidden_dim, n_embd), gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, num_agents) for _ in range(n_block)])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(n_embd),
            _init_linear(nn.Linear(n_embd, hidden_dim)),
        )
        self.agent_id_embedding = nn.Parameter(torch.zeros(1, num_agents, n_embd))
        nn.init.normal_(self.agent_id_embedding, mean=0.0, std=0.02)
        _zero_last_linear(self.output_proj)

    def forward(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(agent_tokens) + self.agent_id_embedding
        x = self.blocks(x)
        return self.output_proj(x)


class DICGStateAdapter(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.core = DICGCore(
            input_dim=hidden_dim,
            encoder_hidden_sizes=(128, 128),
            embedding_dim=128,
            attention_type="general",
            n_gcn_layers=2,
            gcn_bias=True,
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, hidden_dim),
        )
        _zero_last_linear(self.output_proj)

    def forward(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        embeddings_collection, _ = self.core(agent_tokens)
        merged = embeddings_collection[0] + embeddings_collection[-1]
        return self.output_proj(merged)


class TGCNetStateAdapter(nn.Module):
    def __init__(self, hidden_dim: int, num_agents: int):
        super().__init__()
        self.local_encoder = nn.Linear(hidden_dim, 128)
        self.comm = TGCCommunicationBlock(
            hidden_dim=128,
            n_agents=num_agents,
            enc_att_heads=2,
            dec_att_heads=4,
            att_enc_dim=128,
            att_dec_dim=32,
            num_layers=2,
            dropout=0.0,
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, hidden_dim),
        )
        _zero_last_linear(self.output_proj)

    def forward(self, agent_tokens: torch.Tensor) -> torch.Tensor:
        local_hidden = torch.relu(self.local_encoder(agent_tokens))
        decoded_hidden, _ = self.comm(local_hidden)
        merged = torch.cat([local_hidden, decoded_hidden], dim=-1)
        return self.output_proj(merged)


def build_actor_only_optimizer(args, agent: nn.Module) -> torch.optim.Optimizer:
    params: List[torch.nn.Parameter] = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        [
            {
                "params": params,
                "lr": args.head_learning_rate,
                "group_name": "actor_head",
            }
        ],
        eps=1e-5,
        weight_decay=args.weight_decay,
    )


class FrozenMixedBackboneAdapterAgent(MixedTinyVLAAdapterSmolVLASFTAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.vla_agent_name not in self.agent_names or self.smolvla_agent_name not in self.agent_names:
            raise RuntimeError("FrozenMixedBackboneAdapterAgent expects the standard mixed agent ordering")
        for parameter in self.parameters():
            parameter.requires_grad = False

    def _mixed_residuals(
        self,
        vla_state_feature: torch.Tensor,
        smolvla_state_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _adapt_actor_state_features(
        self,
        vla_state_feature: torch.Tensor,
        smolvla_state_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_vla, delta_smol = self._mixed_residuals(vla_state_feature, smolvla_state_feature)
        return vla_state_feature + delta_vla, smolvla_state_feature + delta_smol

    @torch.no_grad()
    def update_state_stats(
        self,
        obs,
        *,
        update_actor: bool = True,
        update_critic: bool = True,
    ) -> None:
        del obs, update_actor, update_critic
        return None

    def freeze_state_stats(self) -> None:
        return None

    def unfreeze_state_stats(self) -> None:
        return None
