import torch
import torch.nn as nn
from torch.distributions import Normal

from train.reinforcement_learning.utils import RunningMeanStd
from train.toy_cnn.model import PlainConv, make_mlp, make_mlp_with_orth_init


class HeteroMAPPOAgent(nn.Module):
    ACTOR_MEAN_CLAMP = 20.0
    ACTOR_LOGSTD_MIN = -5.0
    ACTOR_LOGSTD_MAX = 2.0

    def __init__(
        self,
        agent_names,
        state_dims,
        global_state_dim,
        action_dims,
        camera_count=1,
        normalize_state=True,
    ):
        super().__init__()

        self.agent_names = list(agent_names)
        self.state_dims = dict(state_dims)
        self.action_dims = dict(action_dims)
        self.global_state_dim = int(global_state_dim)

        self.rgb_encoder = PlainConv(
            in_channels=3 * camera_count,
            out_dim=256,
            max_pooling=False,
            inactivated_output=False,
        )

        if normalize_state:
            self.actor_state_rms = nn.ModuleDict(
                {
                    name: RunningMeanStd(shape=(self.state_dims[name],))
                    for name in self.agent_names
                }
            )
            self.critic_state_rms = RunningMeanStd(shape=(global_state_dim,))
        else:
            self.actor_state_rms = None
            self.critic_state_rms = None

        self.actor_state_encoders = nn.ModuleDict(
            {
                name: make_mlp(self.state_dims[name], [256, 256], last_act=False)
                for name in self.agent_names
            }
        )
        self.actor_heads = nn.ModuleDict(
            {
                name: make_mlp_with_orth_init(
                    256 + 256,
                    [512, self.action_dims[name]],
                    last_act=False,
                    is_actor=True,
                )
                for name in self.agent_names
            }
        )
        self.actor_logstd = nn.ParameterDict(
            {
                name: nn.Parameter(torch.ones(1, self.action_dims[name]) * -0.5)
                for name in self.agent_names
            }
        )

        self.critic_state_encoder = make_mlp(
            global_state_dim,
            [512, 512],
            last_act=False,
        )
        self.critic = make_mlp_with_orth_init(
            256 + 512,
            [512, 1],
            last_act=False,
        )

    def _normalize_actor_state(self, name, state):
        if self.actor_state_rms is None:
            return state
        return self.actor_state_rms[name](state)

    def _normalize_global_state(self, global_state):
        if self.critic_state_rms is None:
            return global_state
        return self.critic_state_rms(global_state)

    def _actor_feature(self, name, rgb_feat, agent_state):
        state_feat = self.actor_state_encoders[name](agent_state)
        return torch.cat([rgb_feat, state_feat], dim=1)

    def _critic_feature(self, rgb_feat, global_state):
        global_feat = self.critic_state_encoder(global_state)
        return torch.cat([rgb_feat, global_feat], dim=1)

    def _safe_distribution(self, mean: torch.Tensor, logstd: torch.Tensor):
        mean = torch.nan_to_num(
            mean.float(),
            nan=0.0,
            posinf=self.ACTOR_MEAN_CLAMP,
            neginf=-self.ACTOR_MEAN_CLAMP,
        ).clamp(-self.ACTOR_MEAN_CLAMP, self.ACTOR_MEAN_CLAMP)
        logstd = torch.nan_to_num(
            logstd.float(),
            nan=0.0,
            posinf=self.ACTOR_LOGSTD_MAX,
            neginf=self.ACTOR_LOGSTD_MIN,
        ).clamp(self.ACTOR_LOGSTD_MIN, self.ACTOR_LOGSTD_MAX)
        std = torch.exp(logstd)
        return mean, std

    def _safe_action(self, action, *, mean_ref: torch.Tensor):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=mean_ref.dtype, device=mean_ref.device)
        else:
            action = action.to(device=mean_ref.device, dtype=mean_ref.dtype)
        return torch.nan_to_num(
            action.float(),
            nan=0.0,
            posinf=self.ACTOR_MEAN_CLAMP,
            neginf=-self.ACTOR_MEAN_CLAMP,
        )

    def get_action_and_value(self, batch, actions_input=None, return_token_logits=False):
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
            actor_feat = self._actor_feature(name, rgb_feat, agent_state)
            mean = self.actor_heads[name](actor_feat)
            logstd = self.actor_logstd[name].expand_as(mean)
            mean, std = self._safe_distribution(mean, logstd)
            dist = Normal(mean, std)

            if actions_input is None:
                action = dist.sample()
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
        rgb = batch["rgb"]
        rgb_feat = self.rgb_encoder(rgb)
        actions = {}
        for name in self.agent_names:
            state = self._normalize_actor_state(name, batch[f"agent_states_{name}"])
            feat = self._actor_feature(name, rgb_feat, state)
            mean = self.actor_heads[name](feat)
            mean, std = self._safe_distribution(mean, self.actor_logstd[name].expand_as(mean))
            if deterministic:
                actions[name] = mean
            else:
                actions[name] = Normal(mean, std).sample()
        return actions

    def get_value(self, batch):
        rgb = batch["rgb"]
        global_state = self._normalize_global_state(batch["global_state"])
        rgb_feat = self.rgb_encoder(rgb)
        critic_feat = self._critic_feature(rgb_feat, global_state)
        return self.critic(critic_feat).squeeze(-1)

    @torch.no_grad()
    def update_state_stats(self, obs):
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].update(obs[f"agent_states_{name}"])
        if self.critic_state_rms is not None:
            self.critic_state_rms.update(obs["global_state"])
