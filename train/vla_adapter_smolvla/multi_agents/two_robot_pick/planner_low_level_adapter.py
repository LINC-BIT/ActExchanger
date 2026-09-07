from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from train.vla_adapter_smolvla.multi_agents.two_robot_pick.mixed_sft_agent import (
    MixedTinyVLAAdapterSmolVLASFTAgent,
)
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.model import (
    MultiAgentVLAAdapterMAPPOAgent,
    extract_planner_subtasks_from_batch,
)


class PlannerConditionedMixedAgent(MixedTinyVLAAdapterSmolVLASFTAgent):
    def _extract_planner_subtasks(
        self,
        batch: Mapping[str, Any],
        batch_size: int,
    ) -> Optional[Dict[str, List[Optional[str]]]]:
        return extract_planner_subtasks_from_batch(batch, self.agent_names, batch_size)

    def get_action_and_value(
        self,
        batch: Dict[str, Any],
        actions_input=None,
        return_token_logits: bool = False,
        **kwargs,
    ):
        action_bins_input = kwargs.get("action_bins_input")
        rgb = batch["rgb"]
        vla_state = self._normalize_state(
            batch[f"agent_states_{self.vla_agent_name}"].to(device=self.device, dtype=torch.float32),
            self.vla_agent_name,
        )
        smol_state = self._normalize_state(
            batch[f"agent_states_{self.smolvla_agent_name}"].to(device=self.device, dtype=torch.float32),
            self.smolvla_agent_name,
        )
        planner_subtasks = self._extract_planner_subtasks(batch, vla_state.shape[0])
        vla_planner_subtasks = None if planner_subtasks is None else planner_subtasks[self.vla_agent_name]
        smol_planner_subtasks = None if planner_subtasks is None else planner_subtasks[self.smolvla_agent_name]
        vla_state_feature = self.vla_actor.state_projector(vla_state)
        smol_state_feature = self.smolvla_actor.state_projector(smol_state)
        vla_state_feature, smol_state_feature = self._adapt_actor_state_features(
            vla_state_feature,
            smol_state_feature,
        )

        vla_bins = None
        if action_bins_input is not None and self.vla_agent_name in action_bins_input:
            vla_bins = action_bins_input[self.vla_agent_name]
        elif actions_input is not None and self.vla_agent_name in actions_input:
            vla_bins = self.vla_actor.env_actions_to_bin_indices(actions_input[self.vla_agent_name])
        vla_out = self.vla_actor.get_action_and_stats(
            rgbs=rgb,
            states=vla_state,
            state_features=vla_state_feature,
            action_bins=vla_bins,
            prompt_role_ids=torch.zeros(vla_state.shape[0], device=self.device, dtype=torch.long),
            planner_subtasks=vla_planner_subtasks,
            deterministic=False,
        )

        smol_actions_input = None
        if actions_input is not None and self.smolvla_agent_name in actions_input:
            smol_actions_input = actions_input[self.smolvla_agent_name]
        smol_out = self.smolvla_actor.sample_actions(
            rgbs=rgb,
            states=smol_state,
            state_features=smol_state_feature,
            actions_input=smol_actions_input,
            deterministic=False,
            action_position_placeholders=self.smolvla_actor_action_position_placeholders,
            action_position_actor_placeholders=self.smolvla_actor_action_position_actor_placeholders,
            planner_subtasks=smol_planner_subtasks,
        )

        actions_out = {
            self.vla_agent_name: vla_out["env_actions"].detach().cpu().numpy(),
            self.smolvla_agent_name: smol_out["actions"].detach().cpu().numpy(),
        }
        log_probs = {
            self.vla_agent_name: vla_out["log_prob"],
            self.smolvla_agent_name: smol_out["log_prob"],
        }
        entropies = {
            self.vla_agent_name: vla_out["entropy"],
            self.smolvla_agent_name: smol_out["entropy"],
        }
        value = self._compute_value(batch)

        if return_token_logits:
            return actions_out, log_probs, entropies, value, {self.vla_agent_name: vla_out["token_logits"]}
        return actions_out, log_probs, entropies, value

    @torch.no_grad()
    def get_action(self, batch: Dict[str, Any], deterministic: bool = False):
        rgb = batch["rgb"]
        vla_state = self._normalize_state(
            batch[f"agent_states_{self.vla_agent_name}"].to(device=self.device, dtype=torch.float32),
            self.vla_agent_name,
        )
        smol_state = self._normalize_state(
            batch[f"agent_states_{self.smolvla_agent_name}"].to(device=self.device, dtype=torch.float32),
            self.smolvla_agent_name,
        )
        planner_subtasks = self._extract_planner_subtasks(batch, vla_state.shape[0])
        vla_planner_subtasks = None if planner_subtasks is None else planner_subtasks[self.vla_agent_name]
        smol_planner_subtasks = None if planner_subtasks is None else planner_subtasks[self.smolvla_agent_name]
        vla_state_feature = self.vla_actor.state_projector(vla_state)
        smol_state_feature = self.smolvla_actor.state_projector(smol_state)
        vla_state_feature, smol_state_feature = self._adapt_actor_state_features(
            vla_state_feature,
            smol_state_feature,
        )
        vla_out = self.vla_actor.get_action_and_stats(
            rgbs=rgb,
            states=vla_state,
            state_features=vla_state_feature,
            prompt_role_ids=torch.zeros(vla_state.shape[0], device=self.device, dtype=torch.long),
            planner_subtasks=vla_planner_subtasks,
            deterministic=deterministic,
        )
        smol_out = self.smolvla_actor.sample_actions(
            rgbs=rgb,
            states=smol_state,
            state_features=smol_state_feature,
            deterministic=deterministic,
            action_position_placeholders=self.smolvla_actor_action_position_placeholders,
            action_position_actor_placeholders=self.smolvla_actor_action_position_actor_placeholders,
            planner_subtasks=smol_planner_subtasks,
        )
        return {
            self.vla_agent_name: vla_out["env_actions"].detach().cpu().numpy(),
            self.smolvla_agent_name: smol_out["actions"].detach().cpu().numpy(),
        }


class PlannerConditionedOpenVLAAgent(MultiAgentVLAAdapterMAPPOAgent):
    pass
