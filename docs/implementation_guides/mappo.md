# Adding MAPPO

This guide follows the integration sequence in the
[HAPPO example](../../Readme.md#322-example-adding-happo).

## 1. Agent Model Selection

Select an `AgentModelInterface`. It supplies observation conversion,
action/value inference, and optimizer construction.

```python
from api.marl_method_interface import AgentModelInterface

agent_model: AgentModelInterface = vla_adapter
```

## 2. Actor and Critic Binding

MAPPO uses the selected model's decentralized actors and centralized critic. It
does not require an additional planner.

```python
policy = agent_model.build_policy(device=device, config=config)
batch = agent_model.build_batch_from_obs(obs, device=device)
actions, logprobs, entropies, values = agent_model.get_action_and_value(
    policy, batch
)
```

## 3. Communication Interface

MAPPO has no method-specific message encoder. Communication already implemented
inside the selected agent model remains available.

```python
planner = None
communication = None
```

## 4. Continual Online RL Interface

Bind the implemented rollout and update callbacks to the common Online RL entry.

```python
from api.marl_online_rl_interface import CallbackMARLOnlineRL, run_continual_online_rl
from train.marl.mappo.base import collect_rollout, mappo_update_on_policy
from workload.online_driver import run_online_training

marl_method = CallbackMARLOnlineRL(
    algorithm_name="mappo",
    model_name=agent_model.model_name,
    collect_rollout_fn=collect_rollout,
    update_fn=mappo_update_on_policy,
)
run_continual_online_rl(
    args,
    agent_model=agent_model,
    marl_method=marl_method,
    driver=run_online_training,
)
```
