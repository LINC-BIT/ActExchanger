import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.mappo_pretrain as mappo_pretrain

from train.marl.mat import mat_update_on_policy
from train.marl.mat.model import DecodeBlock, EncodeBlock, _init_linear
from train.toy_cnn.model import make_mlp_with_orth_init
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.toy_cnn_agent import (
    PassObjectToyCNNAgent,
)


def _infer_last_linear_out_dim(module: nn.Module, default: int) -> int:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            return int(child.out_features)
    return int(default)


class ResidualMATPolicy(nn.Module):
    def __init__(
        self,
        *,
        agent_names,
        state_dims,
        global_state_dim,
        action_dims,
        normalize_state: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.backbone_agent = PassObjectToyCNNAgent(
            agent_names=agent_names,
            state_dims=state_dims,
            global_state_dim=global_state_dim,
            action_dims=action_dims,
            normalize_state=normalize_state,
            **kwargs,
        )
        self.agent_names = list(self.backbone_agent.agent_names)
        self.action_dims = dict(action_dims)

        for parameter in self.backbone_agent.parameters():
            parameter.requires_grad = False
        self.backbone_agent.eval()

        state_hidden_dim = _infer_last_linear_out_dim(
            self.backbone_agent.actor_state_encoders[self.agent_names[0]],
            256,
        )
        global_hidden_dim = _infer_last_linear_out_dim(self.backbone_agent.critic_state_encoder, 512)
        rgb_hidden_dim = 256
        n_embd = 256
        n_head = 4
        n_block = 2

        token_dim = rgb_hidden_dim + state_hidden_dim + global_hidden_dim
        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(token_dim),
            _init_linear(nn.Linear(token_dim, n_embd), gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
        )
        self.encoder_ln = nn.LayerNorm(n_embd)
        self.encoder_blocks = nn.Sequential(
            *[EncodeBlock(n_embd, n_head, len(self.agent_names)) for _ in range(n_block)]
        )
        self.action_encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    _init_linear(nn.Linear(int(self.action_dims[name]), n_embd), gain=nn.init.calculate_gain("relu")),
                    nn.GELU(),
                )
                for name in self.agent_names
            }
        )
        self.decoder_ln = nn.LayerNorm(n_embd)
        self.decoder_blocks = nn.ModuleList(
            [DecodeBlock(n_embd, n_head, len(self.agent_names)) for _ in range(n_block)]
        )
        self.actor_residual_heads = nn.ModuleDict(
            {
                name: make_mlp_with_orth_init(
                    n_embd,
                    [n_embd, int(self.action_dims[name])],
                    last_act=False,
                    is_actor=True,
                )
                for name in self.agent_names
            }
        )
        self.critic = make_mlp_with_orth_init(
            n_embd,
            [n_embd, 1],
            last_act=False,
        )
        self.agent_id_embedding = nn.Parameter(torch.zeros(1, len(self.agent_names), n_embd))
        nn.init.normal_(self.agent_id_embedding, mean=0.0, std=0.02)
        self._assert_backbone_frozen()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone_agent.eval()
        self._assert_backbone_frozen()
        return self

    def _assert_backbone_frozen(self) -> None:
        for name, parameter in self.backbone_agent.named_parameters():
            if parameter.requires_grad:
                raise RuntimeError(f"Backbone agent parameter unexpectedly trainable: {name}")

    def _prepare_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        return self.backbone_agent._prepare_batch(batch)

    def _encode_backbone(self, batch: Mapping[str, Any]):
        batch = self._prepare_batch(batch)
        with torch.no_grad():
            rgb = batch["rgb"]
            global_state = batch["global_state"]
            rgb_feat = self.backbone_agent.rgb_encoder(rgb)
            critic_global_state = self.backbone_agent._normalize_global_state(global_state)
            global_feat = self.backbone_agent.critic_state_encoder(critic_global_state)
            critic_feat = self.backbone_agent._critic_feature(rgb_feat, critic_global_state)
            base_value = self.backbone_agent.critic(critic_feat).squeeze(-1)

            token_list = []
            base_means = []
            for name in self.agent_names:
                actor_state = self.backbone_agent._normalize_actor_state(name, batch[f"agent_states_{name}"])
                state_feat = self.backbone_agent.actor_state_encoders[name](actor_state)
                actor_feat = self.backbone_agent._aggregate_actor_feature(name, rgb_feat, actor_state)
                base_mean = self.backbone_agent.actor_heads[name](actor_feat)
                token_list.append(torch.cat([rgb_feat, state_feat, global_feat], dim=-1))
                base_means.append(base_mean)

        obs_tokens = torch.stack(token_list, dim=1)
        obs_emb = self.obs_encoder(obs_tokens) + self.agent_id_embedding
        obs_rep = self.encoder_blocks(self.encoder_ln(obs_emb))
        base_mean_dict = {name: mean for name, mean in zip(self.agent_names, base_means)}
        return obs_rep, base_mean_dict, base_value

    def _decode_hidden(self, shifted_action_emb: torch.Tensor, obs_rep: torch.Tensor) -> torch.Tensor:
        x = shifted_action_emb + self.agent_id_embedding
        x = self.decoder_ln(x)
        for block in self.decoder_blocks:
            x = block(x, obs_rep)
        return x

    def _coerce_action_tensor(self, name: str, action, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=dtype, device=device)
        else:
            action = action.to(device=device, dtype=dtype)
        return torch.nan_to_num(action, nan=0.0, posinf=20.0, neginf=-20.0)

    def _build_shifted_action_embeddings(self, actions_input, *, batch_size: int, device: torch.device, dtype: torch.dtype):
        shifted_action_emb = torch.zeros(
            batch_size,
            len(self.agent_names),
            self.agent_id_embedding.shape[-1],
            device=device,
            dtype=dtype,
        )
        for idx in range(1, len(self.agent_names)):
            prev_name = self.agent_names[idx - 1]
            prev_action = self._coerce_action_tensor(prev_name, actions_input[prev_name], device=device, dtype=dtype)
            shifted_action_emb[:, idx, :] = self.action_encoders[prev_name](prev_action)
        return shifted_action_emb

    def _autoregressive_act(
        self,
        obs_rep: torch.Tensor,
        base_means: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ):
        batch_size = obs_rep.shape[0]
        shifted_action_emb = torch.zeros(
            batch_size,
            len(self.agent_names),
            self.agent_id_embedding.shape[-1],
            device=obs_rep.device,
            dtype=obs_rep.dtype,
        )

        actions_out = {}
        log_probs = {}
        entropies = {}
        for idx, name in enumerate(self.agent_names):
            hidden = self._decode_hidden(shifted_action_emb, obs_rep)
            residual = self.actor_residual_heads[name](hidden[:, idx, :])
            mean = base_means[name] + residual
            logstd = self.backbone_agent.actor_logstd[name].expand_as(mean)
            mean, std = self.backbone_agent._safe_distribution(mean, logstd)
            dist = Normal(mean, std)
            action = mean if deterministic else dist.sample()
            actions_out[name] = action.detach().cpu().numpy()
            log_probs[name] = dist.log_prob(action).sum(-1)
            entropies[name] = dist.entropy().sum(-1)
            if idx + 1 < len(self.agent_names):
                shifted_action_emb[:, idx + 1, :] = self.action_encoders[name](
                    self.backbone_agent._safe_action(action, mean_ref=mean)
                )
        return actions_out, log_probs, entropies

    def _parallel_eval(
        self,
        obs_rep: torch.Tensor,
        base_means: Dict[str, torch.Tensor],
        actions_input,
    ):
        shifted_action_emb = self._build_shifted_action_embeddings(
            actions_input,
            batch_size=obs_rep.shape[0],
            device=obs_rep.device,
            dtype=obs_rep.dtype,
        )
        hidden = self._decode_hidden(shifted_action_emb, obs_rep)

        actions_out = {}
        log_probs = {}
        entropies = {}
        for idx, name in enumerate(self.agent_names):
            residual = self.actor_residual_heads[name](hidden[:, idx, :])
            mean = base_means[name] + residual
            logstd = self.backbone_agent.actor_logstd[name].expand_as(mean)
            mean, std = self.backbone_agent._safe_distribution(mean, logstd)
            dist = Normal(mean, std)
            action = self.backbone_agent._safe_action(actions_input[name], mean_ref=mean)
            actions_out[name] = action.detach().cpu().numpy()
            log_probs[name] = dist.log_prob(action).sum(-1)
            entropies[name] = dist.entropy().sum(-1)
        return actions_out, log_probs, entropies

    def get_trainable_value(self, batch):
        obs_rep, _, base_value = self._encode_backbone(batch)
        pooled_rep = obs_rep.mean(dim=1)
        return base_value + self.critic(pooled_rep).squeeze(-1)

    def get_action_and_value(
        self,
        batch,
        actions_input=None,
        action_bins_input=None,
        deterministic: bool = False,
        return_token_logits: bool = False,
    ):
        if action_bins_input is not None and actions_input is None:
            actions_input = action_bins_input
        obs_rep, base_means, base_value = self._encode_backbone(batch)
        if actions_input is None:
            actions_out, log_probs, entropies = self._autoregressive_act(
                obs_rep,
                base_means,
                deterministic=deterministic,
            )
        else:
            actions_out, log_probs, entropies = self._parallel_eval(obs_rep, base_means, actions_input)

        pooled_rep = obs_rep.mean(dim=1)
        value = base_value + self.critic(pooled_rep).squeeze(-1)
        if return_token_logits:
            return actions_out, log_probs, entropies, value, None
        return actions_out, log_probs, entropies, value

    @torch.no_grad()
    def get_action(self, batch, deterministic: bool = False):
        obs_rep, base_means, _ = self._encode_backbone(batch)
        actions_out, _, _ = self._autoregressive_act(obs_rep, base_means, deterministic=deterministic)
        return {
            name: torch.as_tensor(actions_out[name], device=obs_rep.device, dtype=obs_rep.dtype)
            for name in self.agent_names
        }

    @torch.no_grad()
    def get_value(self, batch):
        obs_rep, _, base_value = self._encode_backbone(batch)
        pooled_rep = obs_rep.mean(dim=1)
        return base_value + self.critic(pooled_rep).squeeze(-1)

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
        self.backbone_agent.freeze_state_stats()

    def unfreeze_state_stats(self) -> None:
        self.backbone_agent.freeze_state_stats()

    def checkpoint_state_dict(self):
        return self.state_dict()

    def load_checkpoint_state_dict(self, state_dict):
        target_state = self.state_dict()
        backbone_target_state = self.backbone_agent.state_dict()
        has_residual_keys = any(
            key.startswith("backbone_agent.")
            or (
                key in target_state
                and key not in backbone_target_state
            )
            for key in state_dict
        )
        if not has_residual_keys:
            self.backbone_agent.load_checkpoint_state_dict(state_dict)
            return

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


def build_optimizer(args, agent):
    actor_params = []
    critic_params = []
    critic_param_ids = {id(parameter) for parameter in agent.critic.parameters() if parameter.requires_grad}
    seen = set()
    for parameter in agent.parameters():
        if not parameter.requires_grad:
            continue
        param_id = id(parameter)
        if param_id in seen:
            continue
        seen.add(param_id)
        if param_id in critic_param_ids:
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
    if critic_params:
        param_groups.append(
            {
                "params": critic_params,
                "lr": args.value_head_learning_rate,
                "group_name": "critic",
            }
        )
    return torch.optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)


def main(args):
    if args.resume_dir is None and not args.init_agent_path:
        raise ValueError("MAT pretrain requires --init-agent-path or --resume-dir")
    args.model_backbone = "toy_cnn"
    args.full_kl_coef = 0.0
    args.log_full_kl = False
    args.critic_warmup_rollouts = 0
    args.actor_warmup_rollouts = 0
    return mappo_pretrain.main(args)


mappo_pretrain.AGENT_CLS = ResidualMATPolicy
mappo_pretrain.OPTIMIZER_FN = build_optimizer
mappo_pretrain.UPDATE_FN = mat_update_on_policy
mappo_pretrain.ALGO_NAME = "mat"
mappo_pretrain.MODEL_NAME = "mixed_five_vla_pass_object_mat"

parse_args = mappo_pretrain.parse_args


if __name__ == "__main__":
    main(parse_args())
