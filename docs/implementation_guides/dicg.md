# Adding DICG

This guide follows the integration sequence in the
[HAPPO example](../../Readme.md#322-example-adding-happo).

## 1. Agent Model Selection

```python
from api.marl_method_interface import AgentModelInterface

agent_model: AgentModelInterface = vla_adapter
agent_names = list(agent_model.agent_names)
```

## 2. Coordination Graph Binding

Construct `DICGAgent` with the workload dimensions. Its actor and critic build
implicit coordination graphs using the order in `agent_names`.

```python
from train.marl.dicg.model import DICGAgent

dicg_policy = DICGAgent(
    agent_names=agent_names,
    state_dim=state_dim,
    action_dim=action_dim,
).to(device)
```

For the five-agent VLA workload, use
[`FiveVLADICGAdapterAgent`](../../train/vla_adapter_smolvla_efficientvla_tinyvla_flower/multi_agents/pass_object_five_robots/baseline_adapters.py).

## 3. Communication Interface

The implicit graph is executed inside the policy, so no external message
adapter is required.

```python
planner = None
communication = None
```

## 4. Continual Online RL Interface

DICG uses the MAPPO-compatible rollout and update implemented by the workload.

```python
from api.marl_online_rl_interface import CallbackMARLOnlineRL, run_continual_online_rl
from train.marl.mappo.base import collect_rollout, mappo_update_on_policy
from workload.online_driver import run_online_training

marl_method = CallbackMARLOnlineRL(
    algorithm_name="dicg",
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
