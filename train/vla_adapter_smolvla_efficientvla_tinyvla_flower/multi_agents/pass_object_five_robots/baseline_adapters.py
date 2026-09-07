from typing import Dict, List

import torch
import torch.nn as nn
from torch.distributions import Normal

from train.marl.dicg.modules import DICGCore
from train.marl.mat.model import EncodeBlock, _init_linear
from train.marl.tgcnet.modules import TGCCommunicationBlock
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.toy_cnn_agent import (
    PassObjectToyCNNAgent,
)


def _zero_last_linear(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
            return


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


class FrozenToyCNNBackboneAdapterAgent(PassObjectToyCNNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.actor_feature_dim = 256

    def enable_critic_training(self) -> None:
        for module in (self.critic_state_encoder, self.critic):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _adapter_residuals(self, state_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _adapt_actor_state_features(self, state_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        residuals = self._adapter_residuals(state_features)
        return {
            name: state_features[name] + residuals[name]
            for name in self.agent_names
        }

    def _normalize_critic_global_state(self, global_state: torch.Tensor) -> torch.Tensor:
        if self.critic_state_rms is not None:
            global_state = self.critic_state_rms(global_state, clip_range=10.0)
        return torch.nan_to_num(global_state, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

    def _compute_value_from_prepared_batch(self, batch) -> torch.Tensor:
        rgb = batch["rgb"]
        global_state = self._normalize_critic_global_state(batch["global_state"])
        rgb_feat = torch.nan_to_num(self.rgb_encoder(rgb), nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        critic_feat = self._critic_feature(rgb_feat, global_state)
        critic_feat = torch.nan_to_num(critic_feat, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(
            self.critic(critic_feat).squeeze(-1),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

    def get_trainable_value(self, batch) -> torch.Tensor:
        batch = self._prepare_batch(batch)
        return self._compute_value_from_prepared_batch(batch)

    def _forward_with_state_features(
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

        rgb = batch["rgb"]
        value = self._compute_value_from_prepared_batch(batch)
        rgb_feat = torch.nan_to_num(self.rgb_encoder(rgb), nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        state_features = {}
        for name in self.agent_names:
            agent_state = torch.nan_to_num(
                self._normalize_actor_state(name, batch[f"agent_states_{name}"]),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            )
            state_features[name] = torch.nan_to_num(
                self.actor_state_encoders[name](agent_state),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            )
        state_features = self._adapt_actor_state_features(state_features)
        state_features = {
            name: torch.nan_to_num(feature, nan=0.0, posinf=1e4, neginf=-1e4)
            for name, feature in state_features.items()
        }

        actions_out = {}
        log_probs = {}
        entropies = {}
        for name in self.agent_names:
            actor_feat = torch.nan_to_num(
                torch.cat([rgb_feat, state_features[name]], dim=1),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            )
            mean = torch.nan_to_num(
                self.actor_heads[name](actor_feat),
                nan=0.0,
                posinf=10.0,
                neginf=-10.0,
            )
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

    def get_action_and_value(
        self,
        batch,
        actions_input=None,
        action_bins_input=None,
        deterministic: bool = False,
        return_token_logits: bool = False,
    ):
        return self._forward_with_state_features(
            batch,
            actions_input=actions_input,
            action_bins_input=action_bins_input,
            deterministic=deterministic,
            return_token_logits=return_token_logits,
        )

    @torch.no_grad()
    def get_action(self, batch, deterministic: bool = False):
        actions_out, _, _, _ = self._forward_with_state_features(batch, deterministic=deterministic)
        return actions_out

    @torch.no_grad()
    def update_state_stats(self, obs, *, update_actor: bool = True, update_critic: bool = True) -> None:
        super().update_state_stats(obs, update_actor=update_actor, update_critic=update_critic)

    def freeze_state_stats(self) -> None:
        super().freeze_state_stats()

    def unfreeze_state_stats(self) -> None:
        super().unfreeze_state_stats()


def build_actor_only_optimizer(args, agent: nn.Module) -> torch.optim.Optimizer:
    params = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        [
            {
                "params": params,
                "lr": args.head_learning_rate,
                "group_name": "actor_adapter",
            }
        ],
        eps=1e-5,
        weight_decay=args.weight_decay,
    )


def build_actor_critic_optimizer(args, agent: nn.Module) -> torch.optim.Optimizer:
    critic_state_param_ids = {
        id(parameter)
        for parameter in agent.critic_state_encoder.parameters()
        if parameter.requires_grad
    }
    critic_param_ids = {
        id(parameter)
        for parameter in agent.critic.parameters()
        if parameter.requires_grad
    }

    actor_params = []
    critic_state_params = []
    critic_params = []
    seen = set()

    for parameter in agent.parameters():
        if not parameter.requires_grad:
            continue
        param_id = id(parameter)
        if param_id in seen:
            continue
        seen.add(param_id)
        if param_id in critic_state_param_ids:
            critic_state_params.append(parameter)
        elif param_id in critic_param_ids:
            critic_params.append(parameter)
        else:
            actor_params.append(parameter)

    param_groups = []
    if actor_params:
        param_groups.append(
            {
                "params": actor_params,
                "lr": args.head_learning_rate,
                "group_name": "actor_adapter",
            }
        )
    if critic_state_params:
        param_groups.append(
            {
                "params": critic_state_params,
                "lr": args.value_head_learning_rate,
                "group_name": "critic_state_encoder",
            }
        )
    if critic_params:
        param_groups.append(
            {
                "params": critic_params,
                "lr": args.value_head_learning_rate,
                "group_name": "critic",
            }
        )
    return torch.optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)


class FiveVLAMATAdapterAgent(FrozenToyCNNBackboneAdapterAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mat_adapter = PerAgentProjectInOutAdapter(
            input_dims=[self.actor_feature_dim for _ in self.agent_names],
            shared_dim=256,
            core=MATStateAdapter(hidden_dim=256, num_agents=len(self.agent_names)),
        )
        self.enable_critic_training()

    def _adapter_residuals(self, state_features):
        outputs = self.mat_adapter([state_features[name] for name in self.agent_names])
        return {name: output for name, output in zip(self.agent_names, outputs)}


class FiveVLADICGAdapterAgent(FrozenToyCNNBackboneAdapterAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dicg_adapter = PerAgentProjectInOutAdapter(
            input_dims=[self.actor_feature_dim for _ in self.agent_names],
            shared_dim=256,
            core=DICGStateAdapter(hidden_dim=256),
        )

    def _adapter_residuals(self, state_features):
        outputs = self.dicg_adapter([state_features[name] for name in self.agent_names])
        return {name: output for name, output in zip(self.agent_names, outputs)}


class FiveVLATGCNetAdapterAgent(FrozenToyCNNBackboneAdapterAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tgcnet_adapter = PerAgentProjectInOutAdapter(
            input_dims=[self.actor_feature_dim for _ in self.agent_names],
            shared_dim=256,
            core=TGCNetStateAdapter(hidden_dim=256, num_agents=len(self.agent_names)),
        )

    def _adapter_residuals(self, state_features):
        outputs = self.tgcnet_adapter([state_features[name] for name in self.agent_names])
        return {name: output for name, output in zip(self.agent_names, outputs)}


FrozenMixedThreeBackboneAdapterAgent = FrozenToyCNNBackboneAdapterAgent
ThreeVLAMATAdapterAgent = FiveVLAMATAdapterAgent
ThreeVLADICGAdapterAgent = FiveVLADICGAdapterAgent
ThreeVLATGCNetAdapterAgent = FiveVLATGCNetAdapterAgent
