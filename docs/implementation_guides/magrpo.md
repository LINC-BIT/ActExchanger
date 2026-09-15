# Adding MAGRPO

This guide follows the integration sequence in the
[RoCo example](../../Readme.md#332-example-adding-roco).

## 1. Agent Model Selection

```python
from api.marl_method_interface import AgentModelInterface

agent_model: AgentModelInterface = vla_adapter
```

## 2. Planner Model Interface

Wrap the workload's high-level subtask planner. MAGRPO retains grouped samples
and computes group-relative advantages during its update.

```python
import torch
from api.marl_method_interface import PlannerModelInterface
from train.vla_adapter_smolvla_efficientvla_tinyvla_flower.multi_agents.pass_object_five_robots.planner_model import (
    HighLevelSubtaskPlannerAgent,
)

class MAGRPOPlannerModel(PlannerModelInterface):
    planner_name = "magrpo_subtask_planner"

    def build_planner(self, *, agent_model, device, config):
        low_level_agent = agent_model.build_policy(device=device, config=config)
        return HighLevelSubtaskPlannerAgent(
            low_level_agent=low_level_agent,
            agent_names=list(agent_model.agent_names),
            state_dim=config["state_dim"],
            state_dims=config.get("state_dims"),
            global_state_dim=config["global_state_dim"],
            action_dim=config["action_dim"],
            action_dims=config.get("action_dims"),
        ).to(device)

    def plan(self, planner, batch, *, deterministic=False):
        return planner.get_action(batch, deterministic=deterministic)

    def get_action_and_value(self, planner, batch, *, actions_input=None, deterministic=False):
        return planner.get_action_and_value(
            batch, actions_input=actions_input, deterministic=deterministic
        )

    def build_optimizer(self, planner, *, config):
        return torch.optim.AdamW(
            [p for p in planner.parameters() if p.requires_grad],
            lr=config.get("planner_learning_rate", 3e-4),
        )

    def update(self, planner, optimizer, rollout, *, config):
        return config["planner_update_fn"](
            planner, optimizer, rollout, config=config
        )

planner_impl = MAGRPOPlannerModel()
planner = planner_impl.build_planner(
    agent_model=agent_model, device=device, config=config
)
```

## 3. Communication Interface

```python
from api.marl_method_interface import CommunicationInterface
from api.marl_method_interface_examples._common import TensorCommunication

class MAGRPOCommunication(CommunicationInterface):
    communication_name = "magrpo_group_message"

    def __init__(self):
        self._impl = TensorCommunication()

    def encode_message(self, sender, receiver, feature, *, action_mask=None):
        return self._impl.encode_message(sender, receiver, feature, action_mask=action_mask)

    def decode_message(self, message, *, receiver, device):
        return self._impl.decode_message(message, receiver=receiver, device=device)

    def aggregate(self, local_feature, remote_features, *, batch=None):
        return self._impl.aggregate(local_feature, remote_features, batch=batch)

    def transmission_size(self, message):
        return self._impl.transmission_size(message)
```

The rollout must retain `group_ids`, agent masks, old log probabilities,
optional reference log probabilities, and shared group rewards.

## 4. Continual Online RL Interface

```python
from api.marl_online_rl_interface import CallbackMARLOnlineRL, run_continual_online_rl
from train.marl.magrpo.base import collect_rollout, magrpo_update_on_policy
from workload.online_driver import run_online_training

marl_method = CallbackMARLOnlineRL(
    algorithm_name="magrpo",
    model_name=agent_model.model_name,
    collect_rollout_fn=collect_rollout,
    update_fn=magrpo_update_on_policy,
)
run_continual_online_rl(
    args, agent_model=agent_model, marl_method=marl_method,
    driver=run_online_training,
)
```
