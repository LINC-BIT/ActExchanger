from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from train.reinforcement_learning.utils import RunningMeanStd
from train.toy_cnn.model import make_mlp_with_orth_init

from .modules import DICGCore


class DICGAgent(nn.Module):
    def __init__(
        self,
        agent_names: Iterable[str],
        state_dim: int,
        action_dim: int,
        *,
        normalize_state: bool = True,
        encoder_hidden_sizes=(128,),
        embedding_dim: int = 64,
        attention_type: str = "general",
        n_gcn_layers: int = 2,
        residual: bool = True,
        gcn_bias: bool = True,
        actor_hidden_sizes=(128, 128),
        critic_hidden_sizes=(128, 128),
        aggregator_type: str = "sum",
        init_logstd: float = -0.5,
    ):
        super().__init__()

        self.agent_names = list(agent_names)
        self.num_agents = len(self.agent_names)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.embedding_dim = int(embedding_dim)
        self.residual = bool(residual)
        self.aggregator_type = str(aggregator_type)
        if self.aggregator_type not in {"sum", "direct"}:
            raise ValueError(f"Unsupported aggregator_type={self.aggregator_type}")

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

        self.actor_core = DICGCore(
            input_dim=self.state_dim,
            encoder_hidden_sizes=encoder_hidden_sizes,
            embedding_dim=self.embedding_dim,
            attention_type=attention_type,
            n_gcn_layers=n_gcn_layers,
            gcn_bias=gcn_bias,
        )
        self.critic_core = DICGCore(
            input_dim=self.state_dim,
            encoder_hidden_sizes=encoder_hidden_sizes,
            embedding_dim=self.embedding_dim,
            attention_type=attention_type,
            n_gcn_layers=n_gcn_layers,
            gcn_bias=gcn_bias,
        )

        self.actor_heads = nn.ModuleDict(
            {
                name: make_mlp_with_orth_init(
                    self.embedding_dim,
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

        if self.aggregator_type == "sum":
            self.critic_head = make_mlp_with_orth_init(
                self.embedding_dim,
                list(critic_hidden_sizes) + [1],
                last_act=False,
            )
        else:
            self.critic_head = make_mlp_with_orth_init(
                self.embedding_dim * self.num_agents,
                list(critic_hidden_sizes) + [1],
                last_act=False,
            )

        self.last_actor_attention_weights: Optional[torch.Tensor] = None
        self.last_critic_attention_weights: Optional[torch.Tensor] = None

    def _stack_agent_states(self, batch: Dict[str, Any], rms_modules=None) -> torch.Tensor:
        per_agent_states = []
        for name in self.agent_names:
            state = batch[f"agent_states_{name}"]
            if rms_modules is not None:
                state = rms_modules[name](state)
            per_agent_states.append(state)
        return torch.stack(per_agent_states, dim=1)

    def _merge_embeddings(self, embeddings_collection):
        if self.residual:
            return embeddings_collection[0] + embeddings_collection[-1]
        return embeddings_collection[-1]

    def _actor_forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        actor_states = self._stack_agent_states(batch, self.actor_state_rms)
        embeddings_collection, attention_weights = self.actor_core(actor_states)
        self.last_actor_attention_weights = attention_weights
        merged_embeddings = self._merge_embeddings(embeddings_collection)
        return {
            name: merged_embeddings[:, idx, :]
            for idx, name in enumerate(self.agent_names)
        }

    def _critic_forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        critic_states = self._stack_agent_states(batch, self.critic_state_rms)
        embeddings_collection, attention_weights = self.critic_core(critic_states)
        self.last_critic_attention_weights = attention_weights
        merged_embeddings = self._merge_embeddings(embeddings_collection)
        if self.aggregator_type == "sum":
            return self.critic_head(merged_embeddings).squeeze(-1).sum(dim=-1)
        return self.critic_head(merged_embeddings.reshape(merged_embeddings.shape[0], -1)).squeeze(-1)

    def get_action_and_value(
        self,
        batch: Dict[str, Any],
        actions_input: Optional[Dict[str, torch.Tensor]] = None,
        return_token_logits: bool = False,
    ):
        actor_embeddings = self._actor_forward(batch)
        value = self._critic_forward(batch)

        actions_out = {}
        log_probs = {}
        entropies = {}
        for name in self.agent_names:
            mean = self.actor_heads[name](actor_embeddings[name])
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
        actor_embeddings = self._actor_forward(batch)
        actions = {}
        for name in self.agent_names:
            mean = self.actor_heads[name](actor_embeddings[name])
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
            raise RuntimeError("No compatible parameters found for DICGAgent")
        merged_state = dict(target_state)
        merged_state.update(matched)
        self.load_state_dict(merged_state, strict=True)
