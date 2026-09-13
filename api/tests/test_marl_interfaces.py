from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from api.marl_method_interface import AgentModelInterface
from api.marl_method_interface_examples.happo_impl import (
    HAPPOAgentModel,
    HAPPOCommunication,
    HAPPOTrainingLoop,
)
from api.marl_method_interface_examples.maple_impl import MAPLEAgentModel, MAPLETrainingLoop
from api.marl_method_interface_examples.roco_impl import (
    ROCOAgentModel,
    ROCOCommunication,
    ROCOPlannerModel,
    ROCOTrainingLoop,
)
from api.marl_online_rl_interface import (
    CallbackMARLOnlineRL,
    MARLMethodSpec,
    MARLOnlineRLInterface,
    run_continual_online_rl,
)
from api.vla_model_interface import VLAAgentSpec, VLAModelInterface


NAMES = ("agent_0", "agent_1")
STATE_DIMS = {name: 4 for name in NAMES}
ACTION_DIMS = {name: 2 for name in NAMES}


def _rollout(implementation, model):
    obs = {"agent_0": torch.randn(3, 4), "agent_1": torch.randn(3, 4)}
    batch = implementation.build_batch_from_obs(obs, device=torch.device("cpu"))
    with torch.no_grad():
        actions, log_probs, _, _ = implementation.get_action_and_value(model, batch)
    return {
        "batch": batch,
        "actions": actions,
        "old_log_probs": log_probs,
        "advantages": torch.tensor([1.0, 0.25, -0.5]),
        "returns": torch.tensor([1.0, 0.5, -0.25]),
    }


def test_happo_performs_sequential_actor_updates():
    implementation = HAPPOAgentModel(NAMES, STATE_DIMS, ACTION_DIMS)
    model = implementation.build_policy(device=torch.device("cpu"), config={"hidden_dim": 16})
    optimizer = implementation.build_optimizer(model, config={"learning_rate": 1e-2})
    rollout = _rollout(implementation, model)
    before = {name: model.actors[name][0].weight.detach().clone() for name in NAMES}

    metrics = HAPPOTrainingLoop().update(model, optimizer, rollout, config={})

    assert set(metrics) >= {"policy_loss/agent_0", "policy_loss/agent_1", "importance_factor", "value_loss"}
    assert all(not torch.equal(before[name], model.actors[name][0].weight) for name in NAMES)


def test_training_loop_runs_callbacks_and_saves_checkpoint(tmp_path: Path):
    implementation = HAPPOAgentModel(NAMES, STATE_DIMS, ACTION_DIMS)
    checkpoint = tmp_path / "loop.pt"

    def factory(**kwargs):
        return kwargs

    def collector(envs, model, **kwargs):
        del envs, kwargs
        return _rollout(implementation, model)

    def evaluator(envs, model, **kwargs):
        del envs, model, kwargs
        return {"success_rate": 0.5}

    result = HAPPOTrainingLoop().run(
        workload="object_stacking",
        config={
            "agent_model_interface": implementation,
            "environment_factory": factory,
            "rollout_collector": collector,
            "evaluator": evaluator,
            "checkpoint_path": checkpoint,
        },
    )
    assert result["evaluation"]["success_rate"] == 0.5
    assert checkpoint.exists()


def test_roco_backend_schema_feedback_and_dialogue_size():
    implementation = ROCOAgentModel(NAMES, STATE_DIMS, ACTION_DIMS)
    model = implementation.build_policy(device=torch.device("cpu"), config={})

    def backend(batch):
        return {
            "dialogue": f"plan {batch['task_text']}",
            "subtasks": {name: "move" for name in NAMES},
            "waypoints": {name: torch.zeros(2, 3) for name in NAMES},
        }

    planner_adapter = ROCOPlannerModel()
    planner = planner_adapter.build_planner(
        agent_model=model,
        device=torch.device("cpu"),
        config={"llm_planner_backend": backend},
    )
    batch = implementation.build_batch_from_obs(
        {"agent_0": torch.zeros(2, 4), "agent_1": torch.ones(2, 4), "task_text": "stack", "collision_feedback": "blocked"},
        device=torch.device("cpu"),
    )
    plan = planner_adapter.plan(planner, batch)
    assert plan["dialogue"] == "plan stack"
    assert plan["requires_replanning"] is True
    assert ROCOTrainingLoop().update(model, None, {"collision_feedback": "blocked"}, config={})["requires_replanning"] == 1.0

    communication = ROCOCommunication()
    message = communication.encode_message("agent_0", "agent_1", torch.ones(2), subtask="move", environment_feedback="blocked")
    assert communication.transmission_size(message) > HAPPOCommunication().transmission_size(message)


def test_maple_updates_future_state_head():
    implementation = MAPLEAgentModel(NAMES, STATE_DIMS, ACTION_DIMS)
    model = implementation.build_policy(device=torch.device("cpu"), config={"hidden_dim": 16})
    optimizer = implementation.build_optimizer(model, config={"learning_rate": 1e-2})
    rollout = _rollout(implementation, model)
    rollout["future_state_target"] = torch.zeros(3, 8)
    before = model.future_state_head.weight.detach().clone()

    metrics = MAPLETrainingLoop().update(model, optimizer, rollout, config={"maple_future_state_coef": 1.0})

    assert metrics["future_state_updated"] == 1.0
    assert metrics["future_state_loss"] >= 0.0
    assert not torch.equal(before, model.future_state_head.weight)


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _TinyVLA(VLAModelInterface):
    model_name = "tiny"
    agent_spec = VLAAgentSpec(NAMES, STATE_DIMS, ACTION_DIMS, 8)
    policy_class = _TinyPolicy

    def build_policy(self, *, device, config):
        return _TinyPolicy().to(device)

    def build_batch_from_obs(self, obs, *, device):
        return {"obs": torch.as_tensor(obs, device=device)}

    def get_action_and_value(self, policy, batch, *, actions_input=None, deterministic=False):
        del actions_input, deterministic
        value = policy.weight.expand(batch["obs"].shape[0])
        return {"agent_0": value}, {}, {}, value

    def get_action(self, policy, batch, *, deterministic=False):
        del deterministic
        return {"agent_0": policy.weight.expand(batch["obs"].shape[0])}

    def get_value(self, policy, batch):
        return policy.weight.expand(batch["obs"].shape[0])

    def configure_trainable_modules(self, policy, *, freeze_vla_backbone):
        policy.weight.requires_grad_(not freeze_vla_backbone)

    def build_optimizer(self, policy, *, config):
        return torch.optim.Adam(policy.parameters(), lr=float(config.get("learning_rate", 1e-3)))


class _TinyMethod(MARLOnlineRLInterface):
    method_spec = MARLMethodSpec("mappo", "tiny_mappo", lambda: None, lambda: {})

    def build_agent(self, *, args, infos, agent_model, device):
        return agent_model.build_policy(device=device, config={**vars(args), **infos})


def test_vla_online_rl_binding_passes_model_specific_builders():
    captured = {}

    def driver(args, **kwargs):
        captured.update(kwargs)
        return "ran"

    vla = _TinyVLA()
    result = run_continual_online_rl(SimpleNamespace(learning_rate=1e-3), agent_model=vla, marl_method=_TinyMethod(), driver=driver)
    assert result == "ran"
    assert captured["collate_fn_builder"](["agent_0"], torch.device("cpu"))(torch.ones(2)).keys() == {"obs"}
    agent = captured["build_agent"](SimpleNamespace(), {}, device=torch.device("cpu"))
    sample = captured["sample_fn_builder"](["agent_0"], agent, torch.device("cpu"), deterministic=True)
    assert sample(torch.ones(2))["agent_0"].shape == (2,)


def test_vla_and_cnn_models_share_the_online_rl_interface():
    implementations = (
        _TinyVLA(),
        HAPPOAgentModel(NAMES, STATE_DIMS, ACTION_DIMS),
    )
    assert isinstance(implementations[0], AgentModelInterface)
    assert implementations[0].agent_names == NAMES

    for implementation in implementations:
        captured = {}

        def driver(args, **kwargs):
            captured.update(kwargs)
            return "ran"

        result = run_continual_online_rl(
            SimpleNamespace(learning_rate=1e-3),
            agent_model=implementation,
            marl_method=_TinyMethod(),
            driver=driver,
        )
        assert result == "ran"
        policy = captured["build_agent"](
            SimpleNamespace(hidden_dim=16),
            {},
            device=torch.device("cpu"),
        )
        assert isinstance(policy, nn.Module)


def test_callback_marl_methods_use_the_shared_online_rl_entry():
    rollout_fn = lambda **kwargs: kwargs
    update_fn = lambda *args, **kwargs: {"loss": 0.0}

    for algorithm_name, rollout_mode in (
        ("happo", "mappo"),
        ("roco", "mappo"),
        ("maple", "maple"),
    ):
        method = CallbackMARLOnlineRL(
            algorithm_name=algorithm_name,
            model_name="tiny",
            collect_rollout_fn=rollout_fn,
            update_fn=update_fn,
            rollout_mode=rollout_mode,
        )
        captured = {}

        def driver(args, **kwargs):
            captured.update(kwargs)
            return "ran"

        assert run_continual_online_rl(
            SimpleNamespace(learning_rate=1e-3),
            agent_model=_TinyVLA(),
            marl_method=method,
            driver=driver,
        ) == "ran"
        assert captured["algo_name"] == algorithm_name
        assert captured["rollout_mode"] == rollout_mode
        assert captured["collect_rollout_fn"] is rollout_fn
        assert captured["update_fn"] is update_fn


def test_vla_checkpoint_round_trip(tmp_path: Path):
    vla = _TinyVLA()
    policy = vla.build_policy(device=torch.device("cpu"), config={})
    checkpoint = tmp_path / "agent.pt"
    vla.save_checkpoint(checkpoint, policy, extra_state={"step": 3})
    payload = torch.load(checkpoint, weights_only=True)
    restored = vla.build_policy(device=torch.device("cpu"), config={})
    restored.weight.data.zero_()
    vla.load_checkpoint_state_dict(restored, payload["policy"])
    assert restored.weight.item() == 1.0
