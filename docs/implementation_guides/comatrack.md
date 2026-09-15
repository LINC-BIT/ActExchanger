# Adding CoMaTrack

This guide follows the integration sequence in the
[MAPLE example](../../Readme.md#342-example-adding-maple).

## 1. Agent Model Selection

Select a `VLAModelInterface` so CoMaTrack receives visual-language features,
per-agent actions, and a centralized value through the common contract.

```python
from api.marl_method_interface import AgentModelInterface

agent_model: AgentModelInterface = vla_adapter
policy = agent_model.build_policy(device=device, config=config)
```

## 2. Communication Interface

Create an adapter for candidate trajectories and tracking features exchanged by
the VLA agents.

```python
from api.marl_method_interface import CommunicationInterface
from api.marl_method_interface_examples._common import TensorCommunication

class CoMaTrackCommunication(CommunicationInterface):
    communication_name = "comatrack_trajectory_message"

    def __init__(self):
        self._impl = TensorCommunication()

    def encode_message(
        self, sender, receiver, feature, *, action_mask=None,
        candidate_trajectory=None, tracking_score=None, group_id=None,
    ):
        message = self._impl.encode_message(
            sender, receiver, feature, action_mask=action_mask
        )
        message.update({
            "candidate_trajectory": candidate_trajectory,
            "tracking_score": tracking_score,
            "group_id": group_id,
        })
        return message

    def decode_message(self, message, *, receiver, device):
        return self._impl.decode_message(message, receiver=receiver, device=device)

    def aggregate(self, local_feature, remote_features, *, batch=None):
        return self._impl.aggregate(local_feature, remote_features, batch=batch)

    def transmission_size(self, message):
        return self._impl.transmission_size(message)

communication = CoMaTrackCommunication()
message = communication.encode_message(
    sender="agent_0",
    receiver="agent_1",
    feature=local_feature,
    candidate_trajectory=candidate_trajectory,
    tracking_score=tracking_score,
    group_id=group_id,
)
received_feature = communication.decode_message(
    message, receiver="agent_1", device=device
)
fused_feature = communication.aggregate(local_feature, [received_feature])
```

## 3. Continual Online RL Interface

Retain candidate trajectories, group IDs, agent masks, tracking scores, old log
probabilities, optional reference log probabilities, and critic values in the
rollout. Then bind the implemented CoMaTrack update.

```python
from api.marl_online_rl_interface import CallbackMARLOnlineRL, run_continual_online_rl
from train.marl.comatrack.base import collect_rollout, comatrack_update_on_policy
from workload.online_driver import run_online_training

marl_method = CallbackMARLOnlineRL(
    algorithm_name="comatrack",
    model_name=agent_model.model_name,
    collect_rollout_fn=collect_rollout,
    update_fn=comatrack_update_on_policy,
)
run_continual_online_rl(
    args, agent_model=agent_model, marl_method=marl_method,
    driver=run_online_training,
)
```

The existing workload entry is
[`comatrack_online_rl.py`](../../train/vla_adapter_smolvla_efficientvla_tinyvla_flower/multi_agents/pass_object_five_robots/comatrack_online_rl.py).
