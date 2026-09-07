from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from train.reinforcement_learning.utils import RunningMeanStd
from train.toy_cnn.model import make_mlp_with_orth_init

from .modules import TGCCommunicationBlock


class TGCNetAgent(nn.Module):
    def __init__(
        self,
        agent_names: Iterable[str],
        state_dim: int,
        action_dim: int,
        *,
        normalize_state: bool = True,
        hidden_dim: int = 128,
        enc_att_heads: int = 2,
        dec_att_heads: int = 4,
        att_enc_dim: int = 128,
        att_dec_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
        actor_hidden_sizes=(128,),
        critic_hidden_sizes=(128, 128),
        init_logstd: float = -0.5,
    ):
        super().__init__()

        self.agent_names = list(agent_names)
        self.num_agents = len(self.agent_names)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        if normalize_state:
            self.actor_state_rms = nn.ModuleDict(
                {name: RunningMeanStd(shape=(self.state_dim,)) for name in self.agent_names}
            )
            self.critic_state_rms = nn.ModuleDict(
                {name: RunningMeanStd(shape=(self.state_dim,)) for name in self.agent_names}
            )
        else:
            self.actor_state_rms = None
            self.critic_state_rms = None

        self.actor_encoder = nn.Linear(self.state_dim, self.hidden_dim)
        self.critic_encoder = nn.Linear(self.state_dim, self.hidden_dim)

        self.actor_comm = TGCCommunicationBlock(
            hidden_dim=self.hidden_dim,
            n_agents=self.num_agents,
            enc_att_heads=enc_att_heads,
            dec_att_heads=dec_att_heads,
            att_enc_dim=att_enc_dim,
            att_dec_dim=att_dec_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.critic_comm = TGCCommunicationBlock(
            hidden_dim=self.hidden_dim,
            n_agents=self.num_agents,
            enc_att_heads=enc_att_heads,
            dec_att_heads=dec_att_heads,
            att_enc_dim=att_enc_dim,
            att_dec_dim=att_dec_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.actor_heads = nn.ModuleDict(
            {
                name: make_mlp_with_orth_init(
                    self.hidden_dim * 2,
                    list(actor_hidden_sizes) + [self.action_dim],
                    last_act=False,
                    is_actor=True,
                )
                for name in self.agent_names
            }
        )
        self.actor_logstd = nn.ParameterDict(
            {name: nn.Parameter(torch.full((1, self.action_dim), float(init_logstd))) for name in self.agent_names}
        )
        self.critic_head = make_mlp_with_orth_init(
            self.hidden_dim * 2,
            list(critic_hidden_sizes) + [1],
            last_act=False,
        )

        self.last_actor_hard_weights: Optional[torch.Tensor] = None
        self.last_critic_hard_weights: Optional[torch.Tensor] = None

    def _stack_agent_states(self, batch: Dict[str, Any], rms_modules=None) -> torch.Tensor:
        per_agent_states = []
        for name in self.agent_names:
            state = batch[f"agent_states_{name}"]
            if rms_modules is not None:
                state = rms_modules[name](state)
            per_agent_states.append(state)
        return torch.stack(per_agent_states, dim=1)

    def _actor_forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        states = self._stack_agent_states(batch, self.actor_state_rms)
        local_hidden = torch.relu(self.actor_encoder(states))
        decoded_hidden, hard_weights = self.actor_comm(local_hidden)
        self.last_actor_hard_weights = hard_weights
        merged_hidden = torch.cat([local_hidden, decoded_hidden], dim=-1)
        return {name: merged_hidden[:, idx, :] for idx, name in enumerate(self.agent_names)}

    def _critic_forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        states = self._stack_agent_states(batch, self.critic_state_rms)
        local_hidden = torch.relu(self.critic_encoder(states))
        decoded_hidden, hard_weights = self.critic_comm(local_hidden)
        self.last_critic_hard_weights = hard_weights
        merged_hidden = torch.cat([local_hidden, decoded_hidden], dim=-1)
        pooled_hidden = merged_hidden.mean(dim=1)
        return self.critic_head(pooled_hidden).squeeze(-1)

    def get_action_and_value(
        self,
        batch: Dict[str, Any],
        actions_input: Optional[Dict[str, torch.Tensor]] = None,
        return_token_logits: bool = False,
    ):
        actor_features = self._actor_forward(batch)
        value = self._critic_forward(batch)

        actions_out = {}
        log_probs = {}
        entropies = {}
        for name in self.agent_names:
            mean = self.actor_heads[name](actor_features[name])
            logstd = self.actor_logstd[name].expand_as(mean)
            std = torch.exp(logstd)
            dist = Normal(mean, std)

            if actions_input is None:
                action = dist.sample()
                actions_out[name] = action.detach().cpu().numpy()
            else:
                action = actions_input[name]
                if not isinstance(action, torch.Tensor):
                    action = torch.as_tensor(action, dtype=mean.dtype, device=mean.device)
                else:
                    action = action.to(device=mean.device, dtype=mean.dtype)
                actions_out[name] = action

            log_probs[name] = dist.log_prob(action).sum(dim=-1)
            entropies[name] = dist.entropy().sum(dim=-1)

        if return_token_logits:
            return actions_out, log_probs, entropies, value, None
        return actions_out, log_probs, entropies, value

    @torch.no_grad()
    def get_action(self, batch: Dict[str, Any], deterministic: bool = False) -> Dict[str, np.ndarray]:
        actor_features = self._actor_forward(batch)
        actions = {}
        for name in self.agent_names:
            mean = self.actor_heads[name](actor_features[name])
            if deterministic:
                action = mean
            else:
                logstd = self.actor_logstd[name].expand_as(mean)
                action = Normal(mean, torch.exp(logstd)).sample()
            actions[name] = action.detach().cpu().numpy()
        return actions

    def get_value(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self._critic_forward(batch)

    @torch.no_grad()
    def update_state_stats(self, obs: Dict[str, Any]) -> None:
        for name in self.agent_names:
            if self.actor_state_rms is not None:
                self.actor_state_rms[name].update(obs[f"agent_states_{name}"])
            if self.critic_state_rms is not None:
                self.critic_state_rms[name].update(obs[f"agent_states_{name}"])

    def freeze_state_stats(self) -> None:
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].freeze()
        if self.critic_state_rms is not None:
            for name in self.agent_names:
                self.critic_state_rms[name].freeze()

    def unfreeze_state_stats(self) -> None:
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].unfreeze()
        if self.critic_state_rms is not None:
            for name in self.agent_names:
                self.critic_state_rms[name].unfreeze()

    def checkpoint_state_dict(self):
        return self.state_dict()

    def load_checkpoint_state_dict(self, state_dict):
        target_state = self.state_dict()
        matched = {}
        for key, value in state_dict.items():
            if key in target_state and target_state[key].shape == value.shape:
                matched[key] = value
        if not matched:
            raise RuntimeError("No compatible parameters found for TGCNetAgent")
        merged_state = dict(target_state)
        merged_state.update(matched)
        self.load_state_dict(merged_state, strict=True)
