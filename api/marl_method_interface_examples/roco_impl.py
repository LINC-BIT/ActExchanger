"""Runnable ROCO-style language-conditioned MARL interface example."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import nn

from api.marl_method_interface import CommunicationInterface, PlannerModelInterface
from api.marl_method_interface_examples.happo_impl import HAPPOAgentModel, HAPPOTrainingLoop
from api.marl_method_interface_examples._common import TensorCommunication


class _RoCoPlanner(nn.Module):
    """Small deterministic stand-in for RoCo's dialogue/planning LLM.

    A production adapter replaces this module with the RoCo language model.  The
    output schema follows the paper: a dialogue turn, sub-task assignments,
    task-space waypoints, and optional environment feedback for replanning.
    """

    def __init__(self, agent_names: Sequence[str], backend: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None):
        super().__init__()
        self.agent_names = tuple(agent_names)
        self.backend = backend
        self.turn = nn.Parameter(torch.zeros(1))

    def forward(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.backend is not None:
            result = dict(self.backend(batch))
            required = {"dialogue", "subtasks", "waypoints"}
            missing = required.difference(result)
            if missing:
                raise ValueError(f"RoCo backend output is missing: {', '.join(sorted(missing))}")
            result.setdefault("requires_replanning", bool(batch.get("collision_feedback")))
            return result
        text = batch.get("task_text", "")
        feedback = batch.get("collision_feedback", "")
        first_state = batch[f"agent_states_{self.agent_names[0]}"]
        batch_size = first_state.shape[0]
        return {
            "dialogue": f"coordinate {self.agent_names} for task: {text}; feedback: {feedback}",
            "subtasks": {name: f"collaborative sub-task for {name}" for name in self.agent_names},
            "waypoints": {
                name: torch.zeros(batch_size, 3, device=self.turn.device)
                for name in self.agent_names
            },
            "requires_replanning": bool(feedback),
        }


class ROCOAgentModel(HAPPOAgentModel):
    """Optional learned low-level controller beneath RoCo's LLM planner."""

    @property
    def model_name(self) -> str:
        return "roco_low_level_controller"

    def build_batch_from_obs(self, obs: Any, *, device: torch.device) -> Mapping[str, Any]:
        batch = dict(super().build_batch_from_obs(obs, device=device))
        task_text = obs.get("task_text", "") if isinstance(obs, Mapping) else ""
        batch["task_text"] = task_text
        if isinstance(obs, Mapping):
            for key in ("collision_feedback", "dialogue_round", "subtasks", "waypoints"):
                if key in obs:
                    batch[key] = obs[key]
        return batch


class ROCOPlannerModel(PlannerModelInterface):
    planner_name = "roco_task_planner"

    def build_planner(self, *, agent_model: nn.Module, device: torch.device, config: Mapping[str, Any]) -> nn.Module:
        return _RoCoPlanner(agent_model.agent_names, backend=config.get("llm_planner_backend")).to(device)

    def plan(self, planner: nn.Module, batch: Mapping[str, Any], *, deterministic: bool = False) -> Mapping[str, Any]:
        del deterministic
        return planner(batch)

    def get_action_and_value(self, planner: nn.Module, batch: Mapping[str, Any], *, actions_input: Optional[Mapping[str, torch.Tensor]] = None, deterministic: bool = False):
        del actions_input, deterministic
        output = planner(batch)
        size = batch[f"agent_states_{planner.agent_names[0]}"].shape[0]
        value = torch.zeros(size, device=next(planner.parameters()).device)
        return output, {}, {}, value

    def build_optimizer(self, planner: nn.Module, *, config: Mapping[str, Any]):
        return torch.optim.Adam(
            planner.parameters(),
            lr=float(config.get("planner_learning_rate", 3e-4)),
        )

    def update(self, planner: nn.Module, optimizer: Any, rollout: Mapping[str, Any], *, config: Mapping[str, Any]) -> Mapping[str, float]:
        del planner, optimizer, config
        feedback = rollout.get("environment_feedback", rollout.get("collision_feedback", ""))
        valid = bool(rollout.get("plan_valid", not bool(feedback)))
        return {
            "plan_valid": float(valid),
            "requires_replanning": float(not valid),
        }


class ROCOCommunication(CommunicationInterface):
    communication_name = "roco_dialogue_message"

    def __init__(self) -> None:
        self._impl = TensorCommunication()

    def encode_message(
        self,
        sender: str,
        receiver: str,
        feature: torch.Tensor,
        *,
        action_mask: Optional[torch.Tensor] = None,
        dialogue_round: int = 0,
        subtask: Optional[str] = None,
        waypoints: Any = None,
        environment_feedback: Optional[str] = None,
    ) -> Any:
        message = self._impl.encode_message(sender, receiver, feature, action_mask=action_mask)
        message.update(
            {
                "dialogue_round": int(dialogue_round),
                "subtask": subtask,
                "waypoints": waypoints,
                "environment_feedback": environment_feedback,
            }
        )
        return message

    def decode_message(self, message: Any, *, receiver: str, device: torch.device) -> torch.Tensor:
        return self._impl.decode_message(message, receiver=receiver, device=device)

    def aggregate(self, local_feature: torch.Tensor, remote_features: Sequence[torch.Tensor], *, batch: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
        return self._impl.aggregate(local_feature, remote_features, batch=batch)

    def transmission_size(self, message: Any) -> int:
        tensor_bytes = self._impl.transmission_size(message)
        text_bytes = sum(
            len(str(message.get(key, "")).encode("utf-8"))
            for key in ("subtask", "environment_feedback", "dialogue_round")
        )
        waypoints = message.get("waypoints")
        if torch.is_tensor(waypoints):
            tensor_bytes += waypoints.numel() * waypoints.element_size()
        return int(tensor_bytes + text_bytes)


class ROCOTrainingLoop(HAPPOTrainingLoop):
    """RoCo orchestration.

    RoCo uses a frozen pretrained LLM and feedback-driven replanning rather than
    policy-gradient training. ``update`` therefore records plan validity and lets
    the next collection turn invoke the planner again when feedback is present.
    """

    def update(self, agent_model: nn.Module, optimizer: Any, rollout: Mapping[str, Any], *, planner: Optional[nn.Module] = None, planner_optimizer: Any = None, config: Mapping[str, Any]) -> Mapping[str, float]:
        del agent_model, optimizer, planner, planner_optimizer, config
        feedback = rollout.get("environment_feedback", rollout.get("collision_feedback", ""))
        valid = bool(rollout.get("plan_valid", not bool(feedback)))
        return {"plan_valid": float(valid), "requires_replanning": float(not valid)}

    def run(self, *, workload: str, config: Mapping[str, Any]) -> Mapping[str, Any]:
        roco_config = dict(config)
        roco_config.setdefault("planner_model_interface", ROCOPlannerModel())
        roco_config.setdefault("communication_interface", ROCOCommunication())
        return super().run(workload=workload, config=roco_config)
