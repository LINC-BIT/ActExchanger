import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from train.reinforcement_learning.utils import RunningMeanStd
from train.toy_cnn.model import PlainConv, make_mlp, make_mlp_with_orth_init


def _init_linear(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)
    return layer


class SelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_agent: int, masked: bool = False):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")

        self.n_head = n_head
        self.masked = masked
        self.key = _init_linear(nn.Linear(n_embd, n_embd))
        self.query = _init_linear(nn.Linear(n_embd, n_embd))
        self.value = _init_linear(nn.Linear(n_embd, n_embd))
        self.proj = _init_linear(nn.Linear(n_embd, n_embd))
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(n_agent, n_agent)).view(1, 1, n_agent, n_agent),
        )

    def forward(self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, emb_dim = query.shape
        head_dim = emb_dim // self.n_head

        k = self.key(key).view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        q = self.query(query).view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        v = self.value(value).view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        if self.masked:
            att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        return self.proj(out)


class EncodeBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_agent: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            _init_linear(nn.Linear(n_embd, n_embd), gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            _init_linear(nn.Linear(n_embd, n_embd)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_agent: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            _init_linear(nn.Linear(n_embd, n_embd), gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            _init_linear(nn.Linear(n_embd, n_embd)),
        )

    def forward(self, x: torch.Tensor, rep_enc: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class MATAgent(nn.Module):
    def __init__(
        self,
        agent_names,
        state_dim,
        global_state_dim,
        action_dim,
        camera_count=1,
        use_depth=False,
        normalize_state=True,
        n_block=2,
        n_embd=256,
        n_head=4,
        action_type="continuous",
    ):
        super().__init__()

        if use_depth:
            raise NotImplementedError("Current MATAgent implementation only supports use_depth=False")

        self.agent_names: List[str] = list(agent_names)
        self.num_agents = len(self.agent_names)
        self.state_dim = state_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type.lower()
        self.is_continuous = self.action_type != "discrete"

        self.rgb_encoder = PlainConv(
            in_channels=3 * camera_count,
            out_dim=256,
            max_pooling=False,
            inactivated_output=False,
        )

        if normalize_state:
            self.actor_state_rms = nn.ModuleDict(
                {name: RunningMeanStd(shape=(state_dim,)) for name in self.agent_names}
            )
            self.critic_state_rms = RunningMeanStd(shape=(global_state_dim,))
        else:
            self.actor_state_rms = None
            self.critic_state_rms = None

        self.actor_state_encoder = make_mlp(state_dim, [256, 256], last_act=False)
        self.critic_state_encoder = make_mlp(global_state_dim, [256, 256], last_act=False)

        token_dim = 256 + 256 + 256
        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(token_dim),
            _init_linear(nn.Linear(token_dim, n_embd), gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
        )
        self.encoder_ln = nn.LayerNorm(n_embd)
        self.encoder_blocks = nn.Sequential(
            *[EncodeBlock(n_embd, n_head, self.num_agents) for _ in range(n_block)]
        )

        if self.is_continuous:
            self.action_encoder = nn.Sequential(
                _init_linear(nn.Linear(action_dim, n_embd), gain=nn.init.calculate_gain("relu")),
                nn.GELU(),
            )
        else:
            self.action_encoder = nn.Sequential(
                _init_linear(nn.Linear(action_dim + 1, n_embd, bias=False), gain=nn.init.calculate_gain("relu")),
                nn.GELU(),
            )

        self.decoder_ln = nn.LayerNorm(n_embd)
        self.decoder_blocks = nn.ModuleList(
            [DecodeBlock(n_embd, n_head, self.num_agents) for _ in range(n_block)]
        )

        self.actor_heads = nn.ModuleDict()
        self.actor_feature_placeholders = nn.ModuleDict()
        self.actor_logstd = nn.ParameterDict()
        for name in self.agent_names:
            self.actor_heads[name] = make_mlp_with_orth_init(
                n_embd,
                [n_embd, action_dim],
                last_act=False,
                is_actor=True,
            )
            self.actor_feature_placeholders[name] = nn.Identity()
            logstd_dim = action_dim if self.is_continuous else 1
            self.actor_logstd[name] = nn.Parameter(torch.ones(1, logstd_dim) * -0.5)

        self.critic = make_mlp_with_orth_init(
            n_embd,
            [n_embd, 1],
            last_act=False,
        )
        self.agent_id_embedding = nn.Parameter(torch.zeros(1, self.num_agents, n_embd))
        nn.init.normal_(self.agent_id_embedding, mean=0.0, std=0.02)

    def _encode_tokens(
        self,
        batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        rgb_feat = self.rgb_encoder(batch["rgb"])
        global_state = batch["global_state"]
        if self.critic_state_rms is not None:
            global_state = self.critic_state_rms(global_state)
        global_feat = self.critic_state_encoder(global_state)

        token_list = []
        actor_states = {}
        for name in self.agent_names:
            actor_state = batch[f"agent_states_{name}"]
            if self.actor_state_rms is not None:
                actor_state = self.actor_state_rms[name](actor_state)
            actor_states[name] = actor_state
            local_feat = self.actor_state_encoder(actor_state)
            token_list.append(torch.cat([rgb_feat, global_feat, local_feat], dim=-1))

        obs_tokens = torch.stack(token_list, dim=1)
        obs_emb = self.obs_encoder(obs_tokens) + self.agent_id_embedding
        obs_rep = self.encoder_blocks(self.encoder_ln(obs_emb))
        pooled_rep = obs_rep.mean(dim=1)
        value = self.critic(pooled_rep).squeeze(-1)
        return obs_rep, value, rgb_feat, actor_states

    def _decode_hidden(self, shifted_actions: torch.Tensor, obs_rep: torch.Tensor) -> torch.Tensor:
        x = self.action_encoder(shifted_actions) + self.agent_id_embedding
        x = self.decoder_ln(x)
        for block in self.decoder_blocks:
            x = block(x, obs_rep)
        return x

    def _stack_continuous_actions(self, actions_input) -> torch.Tensor:
        action_tensors = []
        ref_param = next(self.parameters())
        for name in self.agent_names:
            action = actions_input[name]
            if not isinstance(action, torch.Tensor):
                action = torch.as_tensor(action, dtype=ref_param.dtype, device=ref_param.device)
            else:
                action = action.to(device=ref_param.device, dtype=ref_param.dtype)
            action_tensors.append(action)
        return torch.stack(action_tensors, dim=1)

    def _stack_discrete_actions(self, actions_input) -> torch.Tensor:
        action_tensors = []
        device = next(self.parameters()).device
        for name in self.agent_names:
            action = actions_input[name]
            if not isinstance(action, torch.Tensor):
                action = torch.as_tensor(action, dtype=torch.long, device=device)
            else:
                action = action.to(device=device, dtype=torch.long)
            if action.ndim == 1:
                action = action.unsqueeze(-1)
            action_tensors.append(action)
        return torch.stack(action_tensors, dim=1)

    def _parallel_eval_continuous(
        self,
        obs_rep: torch.Tensor,
        action_tensor: torch.Tensor,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        shifted_actions = torch.zeros_like(action_tensor)
        shifted_actions[:, 1:, :] = action_tensor[:, :-1, :]
        hidden = self._decode_hidden(shifted_actions, obs_rep)

        actions_out = {}
        log_probs = {}
        entropies = {}
        means = {}

        for idx, name in enumerate(self.agent_names):
            feat = self.actor_feature_placeholders[name](hidden[:, idx, :])
            mean = self.actor_heads[name](feat)
            std = torch.exp(self.actor_logstd[name]).expand_as(mean)
            dist = Normal(mean, std)
            action = action_tensor[:, idx, :]
            actions_out[name] = action.detach().cpu().numpy()
            log_probs[name] = dist.log_prob(action).sum(-1)
            entropies[name] = dist.entropy().sum(-1)
            means[name] = mean

        return actions_out, log_probs, entropies, means

    def _autoregressive_act_continuous(
        self,
        obs_rep: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch_size = obs_rep.shape[0]
        device = obs_rep.device
        shifted_actions = torch.zeros(batch_size, self.num_agents, self.action_dim, device=device)

        actions_out = {}
        log_probs = {}
        entropies = {}
        means = {}

        for idx, name in enumerate(self.agent_names):
            hidden = self._decode_hidden(shifted_actions, obs_rep)
            feat = self.actor_feature_placeholders[name](hidden[:, idx, :])
            mean = self.actor_heads[name](feat)
            std = torch.exp(self.actor_logstd[name]).expand_as(mean)
            dist = Normal(mean, std)
            action = mean if deterministic else dist.sample()

            actions_out[name] = action.detach().cpu().numpy()
            log_probs[name] = dist.log_prob(action).sum(-1)
            entropies[name] = dist.entropy().sum(-1)
            means[name] = mean

            if idx + 1 < self.num_agents:
                shifted_actions[:, idx + 1, :] = action

        return actions_out, log_probs, entropies, means

    def _parallel_eval_discrete(
        self,
        obs_rep: torch.Tensor,
        action_tensor: torch.Tensor,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch_size = obs_rep.shape[0]
        device = obs_rep.device
        shifted_actions = torch.zeros(batch_size, self.num_agents, self.action_dim + 1, device=device)
        shifted_actions[:, 0, 0] = 1.0
        one_hot = F.one_hot(action_tensor.squeeze(-1), num_classes=self.action_dim).float()
        shifted_actions[:, 1:, 1:] = one_hot[:, :-1, :]
        hidden = self._decode_hidden(shifted_actions, obs_rep)

        actions_out = {}
        log_probs = {}
        entropies = {}
        logits_dict = {}
        for idx, name in enumerate(self.agent_names):
            feat = self.actor_feature_placeholders[name](hidden[:, idx, :])
            logits = self.actor_heads[name](feat)
            dist = Categorical(logits=logits)
            action = action_tensor[:, idx, 0]
            actions_out[name] = action.unsqueeze(-1).detach().cpu().numpy()
            log_probs[name] = dist.log_prob(action)
            entropies[name] = dist.entropy()
            logits_dict[name] = logits
        return actions_out, log_probs, entropies, logits_dict

    def _autoregressive_act_discrete(
        self,
        obs_rep: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch_size = obs_rep.shape[0]
        device = obs_rep.device
        shifted_actions = torch.zeros(batch_size, self.num_agents, self.action_dim + 1, device=device)
        shifted_actions[:, 0, 0] = 1.0

        actions_out = {}
        log_probs = {}
        entropies = {}
        logits_dict = {}

        for idx, name in enumerate(self.agent_names):
            hidden = self._decode_hidden(shifted_actions, obs_rep)
            feat = self.actor_feature_placeholders[name](hidden[:, idx, :])
            logits = self.actor_heads[name](feat)
            dist = Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else dist.sample()

            actions_out[name] = action.unsqueeze(-1).detach().cpu().numpy()
            log_probs[name] = dist.log_prob(action)
            entropies[name] = dist.entropy()
            logits_dict[name] = logits

            if idx + 1 < self.num_agents:
                shifted_actions[:, idx + 1, 1:] = F.one_hot(action, num_classes=self.action_dim).float()

        return actions_out, log_probs, entropies, logits_dict

    def get_action_and_value(self, batch, actions_input=None):
        obs_rep, value, _, _ = self._encode_tokens(batch)
        if self.is_continuous:
            if actions_input is None:
                actions_out, log_probs, entropies, _ = self._autoregressive_act_continuous(obs_rep)
            else:
                action_tensor = self._stack_continuous_actions(actions_input)
                actions_out, log_probs, entropies, _ = self._parallel_eval_continuous(obs_rep, action_tensor)
        else:
            if actions_input is None:
                actions_out, log_probs, entropies, _ = self._autoregressive_act_discrete(obs_rep)
            else:
                action_tensor = self._stack_discrete_actions(actions_input)
                actions_out, log_probs, entropies, _ = self._parallel_eval_discrete(obs_rep, action_tensor)
        return actions_out, log_probs, entropies, value

    def forward(self, batch):
        obs_rep, value, _, _ = self._encode_tokens(batch)
        if self.is_continuous:
            _, _, _, means = self._autoregressive_act_continuous(obs_rep, deterministic=True)
            outputs = [value.unsqueeze(-1)] + [means[name] for name in self.agent_names]
            return torch.cat(outputs, dim=-1)

        _, _, _, logits_dict = self._autoregressive_act_discrete(obs_rep, deterministic=True)
        outputs = [value.unsqueeze(-1)] + [logits_dict[name] for name in self.agent_names]
        return torch.cat(outputs, dim=-1)

    @torch.no_grad()
    def get_action(self, batch, deterministic=False):
        obs_rep, _, _, _ = self._encode_tokens(batch)
        if self.is_continuous:
            actions_out, _, _, means = self._autoregressive_act_continuous(
                obs_rep,
                deterministic=deterministic,
            )
            if deterministic:
                return {name: means[name] for name in self.agent_names}
            return {
                name: torch.as_tensor(actions_out[name], device=obs_rep.device, dtype=obs_rep.dtype)
                for name in self.agent_names
            }

        actions_out, _, _, _ = self._autoregressive_act_discrete(obs_rep, deterministic=deterministic)
        return {
            name: torch.as_tensor(actions_out[name], device=obs_rep.device, dtype=torch.long)
            for name in self.agent_names
        }

    def get_value(self, batch):
        _, value, _, _ = self._encode_tokens(batch)
        return value

    @torch.no_grad()
    def update_state_stats(self, obs):
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].update(obs[f"agent_states_{name}"])

        if self.critic_state_rms is not None:
            self.critic_state_rms.update(obs["global_state"])

    def reset_logstd(self, new_logstd=-0.5):
        for name in self.agent_names:
            self.actor_logstd[name].data.fill_(new_logstd)

    @torch.no_grad()
    def reset_value_head(self, use_depth=False):
        del use_depth
        device = self.critic[0].weight.device
        self.critic = make_mlp_with_orth_init(
            self.n_embd,
            [self.n_embd, 1],
            last_act=False,
        ).to(device)

    def freeze_state_stats(self):
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].freeze()
        if self.critic_state_rms is not None:
            self.critic_state_rms.freeze()

    def unfreeze_state_stats(self):
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].unfreeze()
        if self.critic_state_rms is not None:
            self.critic_state_rms.unfreeze()
