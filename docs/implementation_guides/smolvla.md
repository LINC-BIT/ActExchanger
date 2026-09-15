# Adding SmolVLA

This guide follows the four steps in the
[VLA-Adapter example](../../Readme.md#312-example-adding-vla-adapter).

## Step 1: Implement `VLAModelInterface`

```python
from pathlib import Path

from api.vla_model_interface import VLAActionOutput, VLAModelInterface
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.mixed_sft_agent import (
    MixedTinyVLAAdapterSmolVLASFTAgent,
)

class SmolVLA(VLAModelInterface):
    def __init__(self, model_dir, agent_spec):
        self.model_dir = Path(model_dir)
        self._agent_spec = agent_spec

    @property
    def model_name(self):
        return "vla_adapter_smolvla"

    @property
    def agent_spec(self):
        return self._agent_spec

    @property
    def policy_class(self):
        return MixedTinyVLAAdapterSmolVLASFTAgent
```

## Step 2: Define the agents and dimensions

```python
from api.vla_model_interface import VLAAgentSpec

spec = VLAAgentSpec(
    agent_names=tuple(agent_names),
    state_dims=state_dims,
    action_dims=action_dims,
    global_state_dim=global_state_dim,
)
```

## Step 3: Bind construction, preprocessing, action generation, and optimization

```python
import inspect
import torch
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.model import (
    build_batch_from_obs as build_workload_batch,
)
```

Add these methods to `SmolVLA`:

```python
    def build_policy(self, *, device, config):
        kwargs = {
            "agent_names": list(self.agent_names),
            "state_dim": max(self.agent_spec.state_dims.values()),
            "state_dims": dict(self.agent_spec.state_dims),
            "global_state_dim": self.agent_spec.global_state_dim,
            "action_dim": max(self.agent_spec.action_dims.values()),
            "action_dims": dict(self.agent_spec.action_dims),
            "model_dir": self.model_dir,
            "freeze_vla_backbone": config.get("freeze_vla_backbone", False),
            **config.get("model_kwargs", {}),
        }
        accepted = inspect.signature(self.policy_class.__init__).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}
        return self.policy_class(**kwargs).to(device)

    def build_batch_from_obs(self, obs, *, device):
        batch = build_workload_batch(obs, list(self.agent_names))
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def generate_actions(
        self, policy, batch, *, actions_input=None, deterministic=False,
        return_value=False, generation_config=None,
    ):
        generation_config = dict(generation_config or {})
        if not return_value and actions_input is None:
            actions = policy.get_action(dict(batch), deterministic=deterministic)
            return VLAActionOutput(actions=actions, auxiliary=generation_config)

        result = policy.get_action_and_value(
            dict(batch),
            actions_input=actions_input,
            deterministic=deterministic,
            **generation_config,
        )
        actions, log_probs, entropies, values, *head_outputs = result
        return VLAActionOutput(
            actions=actions,
            log_probs=log_probs,
            entropies=entropies,
            values=values if return_value else None,
            auxiliary={
                **generation_config,
                "head_outputs": tuple(head_outputs),
            },
        )

    def get_value(self, policy, batch):
        return policy.get_value(dict(batch))

    def configure_trainable_modules(self, policy, *, freeze_vla_backbone):
        for name in ("vla_actor", "smolvla_actor"):
            actor = getattr(policy, name, None)
            if actor is not None:
                actor.configure_trainable_modules(not freeze_vla_backbone)

    def build_optimizer(self, policy, *, config):
        parameters = [p for p in policy.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            parameters, lr=config.get("learning_rate", 3e-5)
        )
```

## Step 4: Instantiate SmolVLA

```python
smolvla = SmolVLA(
    model_dir="ckpt/vla_adapter_smolvla",
    agent_spec=spec,
)
```
