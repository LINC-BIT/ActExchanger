# Adding MAT

This guide follows the integration sequence in the
[HAPPO example](../../Readme.md#322-example-adding-happo).

## 1. Agent Model Selection

```python
from api.marl_method_interface import AgentModelInterface

agent_model: AgentModelInterface = vla_adapter
agent_names = list(agent_model.agent_names)
```

## 2. MAT Policy Binding

Create the MAT policy with the workload dimensions. A VLA-backed workload can
instead wrap its VLA features with the existing `ResidualMATPolicy`.

```python
from train.marl.mat.model import MATAgent

mat_policy = MATAgent(
    agent_names=agent_names,
    state_dim=state_dim,
    global_state_dim=global_state_dim,
    action_dim=action_dim,
).to(device)
```

The five-agent VLA implementation is
[`ResidualMATPolicy`](../../train/vla_adapter_smolvla_efficientvla_tinyvla_flower/multi_agents/pass_object_five_robots/mat_pretrain.py).

## 3. Communication Interface

MAT coordinates agents through its joint Transformer sequence, so it does not
need a separate `CommunicationInterface` object.

```python
planner = None
communication = None
```

## 4. Continual Online RL Interface

```python
from api.marl_online_rl_interface import CallbackMARLOnlineRL, run_continual_online_rl
from train.marl.mat import collect_rollout, mat_update_on_policy
from workload.online_driver import run_online_training

marl_method = CallbackMARLOnlineRL(
    algorithm_name="mat",
    model_name=agent_model.model_name,
    collect_rollout_fn=collect_rollout,
    update_fn=mat_update_on_policy,
)
run_continual_online_rl(
    args,
    agent_model=agent_model,
    marl_method=marl_method,
    driver=run_online_training,
)
```
