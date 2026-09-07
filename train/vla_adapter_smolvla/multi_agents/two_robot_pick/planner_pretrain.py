import os
import sys
import time
from datetime import datetime

sys.path.append(os.getcwd())

from train.toy_cnn.multi_agents.two_robot_pick.gpu_auto_select import configure_cuda_visible_devices

configure_cuda_visible_devices()

import argparse
import random

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from accelerate import Accelerator
from mani_skill.utils.io_utils import dump_json, load_json
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import envs.two_robot_pick_cube_v2  # noqa: F401
from train.internVL.checkpoint_utils import load_agent_checkpoint
from train.reinforcement_learning.evaluate import evaluate
from train.reinforcement_learning.make_env import make_eval_envs
from train.reinforcement_learning.utils import DictArray, compute_gae, get_stage, get_step_infos
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.mixed_sft_agent import (
    MixedTinyVLAAdapterSmolVLASFTAgent,
)
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.model import (
    MultiAgentVLAAdapterMAPPOAgent,
    build_batch_from_obs,
)
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.planner_model import (
    DEFAULT_SUBTASK_VOCAB,
    HighLevelSubtaskPlannerAgent,
)
from train.vla_adapter_smolvla.multi_agents.two_robot_pick.planner_low_level_adapter import (
    PlannerConditionedMixedAgent,
)

mp.set_start_method("spawn", force=True)


MODEL_NAME = "vla_adapter_smolvla_maporl_planner"
ALGO_NAME = "maporl"
planner_update_on_policy = None
build_planner_rollout_meta = None


def resolve_ckpt_dir(args):
    ckpt_task_name = args.ckpt_task_name if args.ckpt_task_name is not None else args.task_name
    task_dir = os.path.join(args.save_dir, f"{ckpt_task_name}/{ALGO_NAME}/{args.robot_name}/{MODEL_NAME}")
    os.makedirs(task_dir, exist_ok=True)
    fresh_run_dir = os.path.join(task_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    resume_dir = args.resume_dir if args.resume_dir is not None else fresh_run_dir
    return {
        "task_dir": task_dir,
        "root_dir": resume_dir,
        "log_dir": os.path.join(resume_dir, "tb"),
        "video_dir": os.path.join(resume_dir, "videos"),
        "latest_agent": os.path.join(resume_dir, "latest_agent.pt"),
        "latest_opt": os.path.join(resume_dir, "latest_opt.pt"),
        "best_agent": os.path.join(resume_dir, "best_agent.pt"),
        "metrics": os.path.join(resume_dir, "metrics.json"),
    }


def make_collate_fn(agent_names):
    def collate_fn(obs):
        return build_batch_from_obs(obs, agent_names)

    return collate_fn


def make_sample_fn(agent_names, agent, deterministic=True):
    collate_fn = make_collate_fn(agent_names)

    def sample_fn(obs):
        batch = collate_fn(obs)
        return agent.get_action(batch, deterministic=deterministic)

    return sample_fn


def flatten_meta(meta):
    flat = {}
    for key, value in meta.items():
        if value is None:
            flat[key] = None
        elif isinstance(value, dict):
            flat[key] = {name: tensor.reshape(-1, *tensor.shape[2:]) for name, tensor in value.items()}
        elif torch.is_tensor(value):
            flat[key] = value.reshape(-1, *value.shape[2:]) if value.ndim >= 2 else value.reshape(-1)
        else:
            flat[key] = value
    return flat


def get_agent_info(args):
    env_kwargs = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "rgb_array",
        "max_episode_steps": args.max_episode_steps,
    }
    test_env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend="gpu",
        env_kwargs=env_kwargs,
    )
    obs, _ = test_env.reset()
    agent_names = list(obs["agent"].keys())
    batch = build_batch_from_obs(obs, agent_names)
    info = {
        "agent_names": agent_names,
        "state_dim": batch[f"agent_states_{agent_names[0]}"].shape[-1],
        "global_state_dim": batch["global_state"].shape[-1],
        "action_dim": test_env.single_action_space[agent_names[0]].shape[0],
    }
    test_env.close()
    return info


def save_checkpoint(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def infer_best_step_from_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        return None, None
    try:
        metrics_log = load_json(metrics_path)
    except Exception as exc:
        print(f"[Resume] failed to read metrics for best-step inference: {exc}")
        return None, None
    best_score = None
    best_step = None
    for row in metrics_log:
        if "score" not in row or "step" not in row:
            continue
        score = float(row["score"])
        step = int(row["step"])
        if best_score is None or score >= best_score:
            best_score = score
            best_step = step
    return best_step, best_score


def infer_best_score_from_metrics_log(metrics_log):
    if not metrics_log:
        return None
    best_score = None
    for row in metrics_log:
        if "score" not in row:
            continue
        score = float(row["score"])
        if best_score is None or score >= best_score:
            best_score = score
    return best_score


def _apply_done_mask(x, mask):
    if torch.is_tensor(x):
        if x.shape[0] == mask.shape[0]:
            return x[mask]
        return x
    if isinstance(x, dict):
        return {k: _apply_done_mask(v, mask) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_apply_done_mask(v, mask) for v in x)
    return x


def collect_planner_rollout(
    args,
    agent,
    collate_fn,
    envs,
    next_obs,
    next_done,
    accelerator,
    writer,
    global_step,
):
    device = accelerator.device
    T = args.rollout_steps
    N = args.num_envs

    obs = DictArray((T, N), envs.single_observation_space, device=device)
    actions = {
        name: torch.zeros((T, N, agent.num_planner_action_components), dtype=torch.long, device=device)
        for name in agent.agent_names
    }
    logprobs = {
        name: torch.zeros((T, N), device=device)
        for name in agent.agent_names
    }
    entropies = {
        name: torch.zeros((T, N), device=device)
        for name in agent.agent_names
    }
    actions = DictArray((T, N), None, data_dict=actions, device=device)
    logprobs = DictArray((T, N), None, data_dict=logprobs, device=device)
    entropies = DictArray((T, N), None, data_dict=entropies, device=device)

    rewards = torch.zeros((T, N), device=device)
    dones = torch.zeros((T, N), device=device)
    values = torch.zeros((T, N), device=device)
    final_values = torch.zeros((T, N), device=device)

    pbar = trange(T, ascii=True, desc="Collecting Planner Rollout")
    now_step = global_step

    for step in pbar:
        now_step += N
        obs[step] = next_obs
        dones[step] = next_done

        batch = collate_fn(next_obs)
        with torch.no_grad():
            env_actions, planner_action_ids, logp_dict, ent_dict, value = agent.rollout_step(batch, deterministic=False)

        values[step] = value.view(N)
        for name in agent.agent_names:
            actions[name][step] = planner_action_ids[name].to(device=device, dtype=torch.long)
            logprobs[name][step] = logp_dict[name].detach().to(device=device)
            entropies[name][step] = ent_dict[name].detach().to(device=device)

        next_obs, reward, terminations, truncations, infos = envs.step(env_actions)
        next_done = torch.logical_or(terminations, truncations).float()
        rewards[step] = reward.view(-1) * args.reward_scale

        if writer is not None and "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            for k, v in final_info["episode"].items():
                writer.add_scalar(f"rollout/{k}", v[done_mask].float().mean(), now_step)

            final_obs = _apply_done_mask(infos["final_observation"], done_mask)
            with torch.no_grad():
                final_batch = collate_fn(final_obs)
                final_value = agent.get_value(final_batch)
                final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = final_value.view(-1)

    return (
        obs,
        actions,
        logprobs,
        rewards,
        dones,
        values,
        final_values,
        next_obs,
        next_done,
    )


def build_low_level_agent(args, infos):
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None
    low_level_agent_path = getattr(args, "low_level_agent_path", None)
    if low_level_agent_path is None:
        low_level_agent_path = args.init_agent_path
    if args.model_backbone == "mixed_tiny_vla_smolvla":
        low_level_agent = PlannerConditionedMixedAgent(
            **infos,
            model_dir=args.model_dir,
            normalize_state=args.normalize_state,
            freeze_vla_backbone=True,
            critic_hidden_dim=args.critic_hidden_dim,
            attention_implementation=args.attn_implementation,
            image_size=args.image_size,
            use_vla_lora=args.use_vla_lora,
            use_vision_lora=args.use_vision_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            train_vision_backbone=args.train_vision_backbone,
            vision_token_pool_size=args.vision_token_pool_size,
            policy_mode=args.policy_mode,
            tiny_hidden_dim=args.tiny_hidden_dim,
            tiny_vision_layers=args.tiny_vision_layers,
            tiny_decoder_layers=args.tiny_decoder_layers,
            tiny_attention_heads=args.tiny_attention_heads,
            tiny_patch_size=args.tiny_patch_size,
            tiny_ffn_mult=args.tiny_ffn_mult,
            tiny_num_action_bins=args.tiny_num_action_bins,
            tiny_prompt_length=args.tiny_prompt_length,
        )
    else:
        if args.model_dir is None:
            raise ValueError("--model-dir is required when model_backbone=smolvla")
        low_level_agent = MultiAgentVLAAdapterMAPPOAgent(
            **infos,
            model_dir=args.model_dir,
            normalize_state=args.normalize_state,
            freeze_vla_backbone=True,
            critic_hidden_dim=args.critic_hidden_dim,
            attention_implementation=args.attn_implementation,
            image_size=args.image_size,
            use_vla_lora=args.use_vla_lora,
            use_vision_lora=args.use_vision_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            train_vision_backbone=args.train_vision_backbone,
            vision_token_pool_size=args.vision_token_pool_size,
            policy_mode=args.policy_mode,
            enable_planner_subtasks=True,
        )
    if low_level_agent_path is None:
        raise ValueError("--low-level-agent-path (or --init-agent-path for pretraining) is required")
    load_agent_checkpoint(low_level_agent, low_level_agent_path, map_location="cpu", label="planner low-level init")
    low_level_agent.freeze_state_stats()
    low_level_agent.eval()
    return low_level_agent


def build_optimizer(args, agent):
    actor_params = (
        list(agent.local_state_encoder.parameters())
        + list(agent.global_state_encoder.parameters())
        + list(agent.coordinator.parameters())
        + list(agent.actor_heads.parameters())
        + [agent.agent_id_embedding]
    )
    critic_params = list(agent.critic.parameters())
    param_groups = []
    if actor_params:
        param_groups.append(
            {
                "params": [p for p in actor_params if p.requires_grad],
                "lr": args.head_learning_rate,
                "group_name": "actor_head",
            }
        )
    if critic_params:
        param_groups.append(
            {
                "params": [p for p in critic_params if p.requires_grad],
                "lr": args.value_head_learning_rate,
                "group_name": "critic",
            }
        )
    return torch.optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)


def maybe_load_initial_agent(args, ckpt, agent):
    if args.resume_dir is not None:
        resume_agent_path = ckpt["best_agent"] if args.resume_use_best_agent else ckpt["latest_agent"]
        if not os.path.exists(resume_agent_path) and args.resume_use_best_agent and os.path.exists(ckpt["latest_agent"]):
            resume_agent_path = ckpt["latest_agent"]
        if os.path.exists(resume_agent_path):
            print(f"[Train] resume planner agent from {resume_agent_path}")
            agent.load_checkpoint_state_dict(torch.load(resume_agent_path, map_location="cpu"))
            best_step = None
            if args.resume_use_best_agent and resume_agent_path == ckpt["best_agent"]:
                best_step, best_score = infer_best_step_from_metrics(ckpt["metrics"])
                if best_step is not None:
                    print(f"[Resume] inferred best checkpoint step={best_step} score={best_score:.4f}")
            return {
                "resumed": True,
                "used_best_agent": bool(args.resume_use_best_agent and resume_agent_path == ckpt["best_agent"]),
                "best_step": best_step,
            }
    return {"resumed": False, "used_best_agent": False, "best_step": None}


def main(args):
    if planner_update_on_policy is None:
        raise RuntimeError("planner_update_on_policy is not configured")
    if getattr(args, "model_dir", None) == "":
        args.model_dir = None

    torch.backends.cudnn.deterministic = not args.ignore_torch_deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator(mixed_precision="bf16" if args.use_amp else "no")
    device = accelerator.device

    ckpt = resolve_ckpt_dir(args)
    step_infos = get_step_infos(args)
    infos = get_agent_info(args)
    agent_names = infos["agent_names"]
    collate_fn = make_collate_fn(agent_names)

    if accelerator.is_main_process:
        writer = SummaryWriter(ckpt["log_dir"])
        print(f"[TensorBoard] Logging to {ckpt['log_dir']}")
    else:
        writer = None

    low_level_agent = build_low_level_agent(args, infos)
    agent = HighLevelSubtaskPlannerAgent(
        low_level_agent=low_level_agent,
        agent_names=agent_names,
        state_dim=infos["state_dim"],
        global_state_dim=infos["global_state_dim"],
        action_dim=infos["action_dim"],
        normalize_state=args.normalize_state,
        planner_hidden_dim=args.planner_hidden_dim,
        planner_layers=args.planner_layers,
        planner_subtasks=DEFAULT_SUBTASK_VOCAB,
        planner_text_mode=args.planner_text_mode,
        low_level_deterministic=args.planner_low_level_deterministic,
    )
    optimizer = build_optimizer(args, agent)

    resume_info = maybe_load_initial_agent(args, ckpt, agent)
    global_steps = 0
    if os.path.exists(ckpt["latest_opt"]):
        latest_opt = torch.load(ckpt["latest_opt"], map_location="cpu")
        if resume_info["resumed"] and resume_info["used_best_agent"]:
            if resume_info["best_step"] is not None:
                global_steps = int(resume_info["best_step"])
            else:
                global_steps = int(latest_opt.get("step", 0))
                if accelerator.is_main_process:
                    print(
                        "[Resume] best_agent resumed without recoverable best optimizer state; "
                        "keeping a fresh optimizer state"
                    )
        else:
            optimizer.load_state_dict(latest_opt["opt"])
            global_steps = int(latest_opt["step"])
    agent.sync_reference_policy()

    agent, optimizer = accelerator.prepare(agent, optimizer)

    env_kwargs = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "none",
        "max_episode_steps": args.max_episode_steps,
        "sim_backend": "physx_cuda",
    }
    env_kwargs_for_eval = {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "rgb_array",
        "max_episode_steps": args.max_episode_steps,
    }

    if args.evaluate_mode and args.eval_agent_dir is None:
        raise ValueError("Please provide --eval-agent-dir for evaluation mode")

    eval_envs = make_eval_envs(
        env_id=args.task_name,
        num_envs=args.num_eval_envs,
        sim_backend="gpu",
        env_kwargs=env_kwargs_for_eval,
        video_dir=(
            os.path.join(args.eval_agent_dir, "videos_eval")
            if args.evaluate_mode
            else f"{ckpt['video_dir']}_train"
        ),
    )

    if args.evaluate_mode:
        unwrapped_agent = accelerator.unwrap_model(agent)
        unwrapped_agent.load_checkpoint_state_dict(
            torch.load(os.path.join(args.eval_agent_dir, "best_agent.pt"), map_location="cpu")
        )
        unwrapped_agent.eval()
        eval_metrics = evaluate(
            n=100,
            sample_fn=make_sample_fn(agent_names, unwrapped_agent, deterministic=True),
            eval_envs=eval_envs,
        )
        payload = {k: float(v.mean()) for k, v in eval_metrics.items()}
        if accelerator.is_main_process:
            for key, value in eval_metrics.items():
                print(f"eval_{key}_mean={value.mean()}")
            dump_json(os.path.join(ckpt["root_dir"], "eval_metrics.json"), payload)
        return

    envs = gym.make(args.task_name, num_envs=args.num_envs, **env_kwargs)
    envs = ManiSkillVectorEnv(
        envs,
        args.num_envs,
        ignore_terminations=args.ignore_partial_reset,
        record_metrics=True,
    )
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    metrics_log = load_json(ckpt["metrics"]) if os.path.exists(ckpt["metrics"]) else []
    best_score = infer_best_score_from_metrics_log(metrics_log)
    if best_score is None:
        best_score = -1.0
    elif accelerator.is_main_process:
        print(f"[Resume] restored historical best score={best_score:.4f} from metrics log")
    start_time = time.time()
    last_save_skip = args.resume_dir is not None

    if args.minibatch_size == 0:
        args.minibatch_size = step_infos["rollot_steps"] // args.num_minibatch // args.grad_accum_steps

    pbar = tqdm(total=step_infos["total_steps"], initial=global_steps, ascii=True)
    init_batch = collate_fn(next_obs)
    agent.unfreeze_state_stats()
    agent.update_state_stats(init_batch)
    agent.freeze_state_stats()

    while global_steps < step_infos["total_steps"]:
        unwrapped_agent = accelerator.unwrap_model(agent)
        unwrapped_agent.eval()

        if not last_save_skip and global_steps % step_infos["save_interval_steps"] == 0 and accelerator.is_main_process:
            save_checkpoint(ckpt["latest_agent"], unwrapped_agent.checkpoint_state_dict())
            save_checkpoint(ckpt["latest_opt"], {"opt": optimizer.state_dict(), "step": global_steps})

            eval_metrics = evaluate(
                n=1,
                sample_fn=make_sample_fn(agent_names, unwrapped_agent, deterministic=True),
                eval_envs=eval_envs,
            )
            for key, value in eval_metrics.items():
                writer.add_scalar(f"eval/{key}", value.mean(), global_steps)
                print(f"eval_{key}_mean={value.mean()}")

            score = eval_metrics.get("success_rate", eval_metrics[list(eval_metrics.keys())[0]]).mean()
            pbar.set_postfix(eval_score=score)
            metrics_log.append({"step": global_steps, "score": float(score)})
            dump_json(ckpt["metrics"], metrics_log)
            if score >= best_score:
                best_score = float(score)
                save_checkpoint(ckpt["best_agent"], unwrapped_agent.checkpoint_state_dict())
                print(f"[Eval] New best planner saved (score={score:.3f})")

        last_save_skip = False

        rollout = collect_planner_rollout(
            args=args,
            agent=agent,
            collate_fn=collate_fn,
            envs=envs,
            next_obs=next_obs,
            next_done=next_done,
            accelerator=accelerator,
            writer=writer,
            global_step=global_steps,
        )
        (
            obs_buf,
            act_buf,
            logp_buf,
            rew_buf,
            done_buf,
            val_buf,
            final_val_buf,
            next_obs,
            next_done,
        ) = rollout

        meta = {}
        if build_planner_rollout_meta is not None:
            meta = build_planner_rollout_meta(
                args,
                agent_names=agent_names,
                rewards=rew_buf,
                dones=done_buf,
            )

        next_batch = collate_fn(next_obs)
        adv_buf, ret_buf = compute_gae(
            rew_buf,
            done_buf,
            val_buf,
            final_val_buf,
            next_batch,
            next_done,
            agent,
            lambda x: x,
            args,
            accelerator,
        )

        data = (
            obs_buf.reshape((-1,)),
            act_buf.reshape((-1,)),
            logp_buf.reshape((-1,)),
            adv_buf.reshape(-1),
            ret_buf.reshape(-1),
            val_buf.reshape(-1),
            flatten_meta(meta) if meta else {},
        )

        agent.train()
        stats = planner_update_on_policy(
            args,
            agent,
            optimizer,
            data,
            collate_fn,
            accelerator,
            get_stage(global_steps, step_infos),
            writer,
            -1,
        )

        agent.unfreeze_state_stats()
        agent.update_state_stats(collate_fn(obs_buf.reshape((-1,))))
        agent.update_state_stats(collate_fn(next_obs))
        agent.freeze_state_stats()

        global_steps += step_infos["rollot_steps"]
        pbar.update(step_infos["rollot_steps"])

        if accelerator.is_main_process:
            y_pred = val_buf.flatten(0, 1).cpu().numpy()
            y_true = ret_buf.flatten(0, 1).cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            sps = global_steps / max(time.time() - start_time, 1e-6)
            writer.add_scalar("charts/SPS", sps, global_steps)
            writer.add_scalar("loss/policy", stats["policy_loss"], global_steps)
            writer.add_scalar("loss/value", stats["value_loss"], global_steps)
            writer.add_scalar("loss/entropy", stats["entropy"], global_steps)
            writer.add_scalar("loss/approx_kl", stats["approx_kl"], global_steps)
            writer.add_scalar("loss/old_approx_kl", stats["old_approx_kl"], global_steps)
            writer.add_scalar("loss/clip_frac", stats["clip_frac"], global_steps)
            writer.add_scalar("loss/explained_var", explained_var, global_steps)

    if accelerator.is_main_process:
        unwrapped_agent = accelerator.unwrap_model(agent)
        save_checkpoint(ckpt["latest_agent"], unwrapped_agent.checkpoint_state_dict())
        save_checkpoint(ckpt["latest_opt"], {"opt": optimizer.state_dict(), "step": global_steps})
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="TwoRobotPickCube-v2")
    parser.add_argument("--ckpt-task-name", type=str, default=None)
    parser.add_argument("--robot-name", type=str, default="pandas_pandas")
    parser.add_argument("--seed", type=int, default=1788)
    parser.add_argument("--total-steps", type=int, default=20_000_000)
    parser.add_argument("--critic-warmup-rollouts", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-eval-envs", type=int, default=16)
    parser.add_argument("--ignore-partial-reset", action="store_true")
    parser.add_argument("--ignore-torch-deterministic", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--num-minibatch", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--normalize-state", action="store_true")
    parser.add_argument("--not-normalize-adv", action="store_true")
    parser.add_argument("--finite-horizon-gae", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.956)
    parser.add_argument("--gae-lambda", type=float, default=0.966)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--value-huber-delta", type=float, default=10.0)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--save-interval-per-rollout", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default="ckpt")
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--resume-use-best-agent", action="store_true")
    parser.add_argument("--init-agent-path", type=str, default=None)
    parser.add_argument("--low-level-agent-path", type=str, default=None)
    parser.add_argument("--evaluate-mode", action="store_true")
    parser.add_argument("--eval-agent-dir", type=str, default=None)
    parser.add_argument("--obs-mode", type=str, default="state_dict+rgb")
    parser.add_argument("--control-mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--reward-mode", type=str, default="normalized_dense")
    parser.add_argument("--max-episode-steps", type=int, default=100)

    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--model-backbone", type=str, default="smolvla", choices=["smolvla", "mixed_tiny_vla_smolvla"])
    parser.add_argument("--freeze-vla-backbone", action="store_true")
    parser.add_argument("--critic-hidden-dim", type=int, default=512)
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--use-vla-lora", action="store_true")
    parser.add_argument("--use-vision-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-vision-backbone", action="store_true")
    parser.add_argument("--vision-token-pool-size", type=int, default=None)
    parser.add_argument("--policy-mode", type=str, default="residual", choices=["residual", "native"])

    parser.add_argument("--tiny-hidden-dim", type=int, default=640)
    parser.add_argument("--tiny-vision-layers", type=int, default=7)
    parser.add_argument("--tiny-decoder-layers", type=int, default=8)
    parser.add_argument("--tiny-attention-heads", type=int, default=10)
    parser.add_argument("--tiny-patch-size", type=int, default=14)
    parser.add_argument("--tiny-ffn-mult", type=int, default=4)
    parser.add_argument("--tiny-num-action-bins", type=int, default=256)
    parser.add_argument("--tiny-prompt-length", type=int, default=24)

    parser.add_argument("--planner-hidden-dim", type=int, default=256)
    parser.add_argument("--planner-layers", type=int, default=2)
    parser.add_argument("--planner-text-mode", type=str, default="fixed", choices=["fixed", "tiny_text"])
    parser.add_argument("--planner-low-level-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--value-head-learning-rate", type=float, default=1e-4)

    parser.add_argument("--maporl-team-adv-coef", type=float, default=0.0)
    parser.add_argument("--maporl-kl-coef", type=float, default=0.0)
    parser.add_argument("--magrpo-eps", type=float, default=1e-5)
    parser.add_argument("--magrpo-gamma", type=float, default=1.0)
    parser.add_argument("--magrpo-group-size", type=int, default=0)
    parser.add_argument("--magrpo-kl-coef", type=float, default=0.0)
    parser.add_argument("--magrpo-ent-coef", type=float, default=0.0)
    parser.add_argument("--mpdf-eps", type=float, default=1e-5)
    parser.add_argument("--mpdf-group-size", type=int, default=0)
    parser.add_argument("--mpdf-rank-temperature", type=float, default=1.0)
    parser.add_argument("--mpdf-kl-coef", type=float, default=0.0)
    parser.add_argument("--mpdf-ent-coef", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
