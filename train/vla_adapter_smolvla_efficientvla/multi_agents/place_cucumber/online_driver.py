import os
import time
import bisect
import random
import inspect
from datetime import datetime
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
from accelerate import Accelerator
from mani_skill.utils.io_utils import dump_json, load_json
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import envs.place_cucumber  # noqa: F401
from train.internVL.checkpoint_utils import load_agent_checkpoint
from train.reinforcement_learning.evaluate import evaluate
from train.reinforcement_learning.make_env import make_eval_envs
from train.reinforcement_learning.utils import compute_gae, get_stage, get_step_infos
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.mappo_feature_aggregator_pretrain import (
    build_clients,
    infer_aggregator_ckpt_prefix,
    iter_client_feature_aggregators,
    log_effective_transmission_stats,
    log_agent_feature_cosine_similarity,
    log_feature_aggregator_agent_cosine_similarity,
    log_uploaded_q_task_stats,
    set_client_feature_aggregator_requires_grad,
)
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.model import build_batch_from_obs
from train.vla_adapter_smolvla_efficientvla.multi_agents.place_cucumber.online_utils import (
    build_continual_env_schedule,
)


DEFAULT_ENV_KWARGS_LIST = [
    {"name": "CucumberSizeSmall70High", "cucumber_size_scale": 0.70},
    {"name": "CucumberSizeSmall50Low", "cucumber_size_scale": 0.50},
    {"name": "CucumberSizeLarge20High", "cucumber_size_scale": 1.20},
    {"name": "CucumberFarOutLow", "cucumber1_xy": (0.18, 0.39), "cucumber2_xy": (0.18, -0.35)},
    {"name": "CucumberSizeLarge25High", "cucumber_size_scale": 1.25},
    {"name": "CucumberSizeSmall70HighRepeat", "cucumber_size_scale": 0.70},
    {"name": "CucumberSizeSmall50LowRepeat", "cucumber_size_scale": 0.50},
    {"name": "CucumberSizeLarge20HighRepeat", "cucumber_size_scale": 1.20},
    {"name": "CucumberFarOutLowRepeat", "cucumber1_xy": (0.18, 0.39), "cucumber2_xy": (0.18, -0.35)},
    {"name": "CucumberSizeLarge25HighRepeat", "cucumber_size_scale": 1.25},
]

FEATURE_AGGREGATOR_DEFAULTS = {
    "feature_aggregator_attention_num_heads": 4,
    "feature_aggregator_gate_type": "two-layers",
    "feature_aggregator_gate_activation": "relu",
    "feature_aggregator_norm_type": "none",
    "feature_aggregator_feature_gate_open_max": 0.20,
    "feature_aggregator_action_gate_open_max": 0.06,
    "feature_aggregator_q_ret_weight": 0.60,
    "feature_aggregator_q_attn_weight": 0.40,
    "feature_aggregator_remote_dropout_prob": 0.0,
    "feature_aggregator_remote_noise_std": 0.0,
    "feature_aggregator_remote_stale_shift_max": 0,
    "feature_selector_topk_trajectories": 4,
    "feature_selector_temporal_pool_steps": 16,
    "feature_selector_strategy": "topk_return",
    "eval_feature_selector_strategy": None,
    "feature_selector_alpha": 0.2,
}


def resolve_ckpt_dir(args, *, algo_name: str, model_name: str):
    ckpt_task_name = args.ckpt_task_name if args.ckpt_task_name is not None else args.task_name
    task_suffix = "online_wo_ag" if getattr(args, "not_train_aggregator", False) else "online_baseline"
    task_dir = os.path.join(args.save_dir, f"{ckpt_task_name}_{task_suffix}/{algo_name}/{args.robot_name}/{model_name}")
    os.makedirs(task_dir, exist_ok=True)

    fresh_run_dir = os.path.join(task_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    resume_dir = args.resume_dir if args.resume_dir is not None else fresh_run_dir
    root_dir = resume_dir
    os.makedirs(root_dir, exist_ok=True)
    return {
        "task_dir": task_dir,
        "root_dir": root_dir,
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
        "state_dims": {name: batch[f"agent_states_{name}"].shape[-1] for name in agent_names},
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


def build_base_env_kwargs(args):
    return {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "rgb_array",
        "max_episode_steps": args.max_episode_steps,
        "sim_backend": "physx_cuda",
    }


def make_envs_for_env_kwargs(args, ckpt, env_kwargs_update_id, base_env_kwargs, env_kwargs_list):
    tmp_env_kwargs = base_env_kwargs.copy()
    env_kwargs_update = env_kwargs_list[env_kwargs_update_id].copy()
    env_name = env_kwargs_update.pop("name")
    tmp_env_kwargs.update(**env_kwargs_update)

    env_kwargs_for_eval = tmp_env_kwargs.copy()
    env_kwargs_for_eval.pop("sim_backend")
    env_kwargs_for_eval["render_mode"] = "rgb_array"
    tmp_env_kwargs["render_mode"] = "none"

    eval_video_root = f"{ckpt['video_dir']}_eval" if args.evaluate_mode else f"{ckpt['video_dir']}_train"
    eval_envs = make_eval_envs(
        env_id=args.task_name,
        num_envs=args.num_eval_envs,
        sim_backend="gpu",
        env_kwargs=env_kwargs_for_eval,
        video_dir=os.path.join(eval_video_root, env_name),
    )
    eval_envs.reset(seed=args.seed)

    envs = gym.make(args.task_name, num_envs=args.num_envs, **tmp_env_kwargs)
    envs = ManiSkillVectorEnv(
        envs,
        args.num_envs,
        ignore_terminations=args.ignore_partial_reset,
        record_metrics=True,
    )
    return envs, eval_envs, env_name


def maybe_load_initial_agent(args, ckpt, agent, init_label="Online init"):
    if args.resume_dir is not None:
        resume_agent_path = ckpt["best_agent"] if args.resume_use_best_agent else ckpt["latest_agent"]
        if not os.path.exists(resume_agent_path) and args.resume_use_best_agent and os.path.exists(ckpt["latest_agent"]):
            print(
                f"[Resume] requested best-agent resume but {resume_agent_path} is missing; "
                f"fall back to latest agent {ckpt['latest_agent']}"
            )
            resume_agent_path = ckpt["latest_agent"]
        if os.path.exists(resume_agent_path):
            print(f"[Train] resume agent from {resume_agent_path}")
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
    if args.init_agent_path:
        load_agent_checkpoint(agent, args.init_agent_path, map_location="cpu", label=init_label)
    return {"resumed": False, "used_best_agent": False, "best_step": None}


def ensure_feature_aggregator_args(args):
    for key, value in FEATURE_AGGREGATOR_DEFAULTS.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)


def load_client_aggregators(clients, load_dir, ckpt_prefix):
    if load_dir is None:
        return
    for name in clients.keys():
        ag_path = os.path.join(load_dir, f"{ckpt_prefix}{name}.pt")
        if os.path.exists(ag_path):
            clients[name].load_feature_aggregators(ag_path)
            print(f"[Checkpoint] loaded feature aggregators for {name}: {ag_path}")
        else:
            print(f"[Checkpoint] missing feature aggregators for {name}: {ag_path}")


def initialize_client_graph(clients, agent, load_dir=None, ckpt_prefix="best_ag_"):
    agent_names = list(clients.keys())
    load_client_aggregators(clients, load_dir, ckpt_prefix)
    client_infos = {name: clients[name].before_training_start(agent) for name in agent_names}
    for recv_name, client_recv in clients.items():
        for send_name, client_send_info in client_infos.items():
            if recv_name != send_name:
                client_recv.add_feature_aggregator(send_name, client_send_info)


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


def run_online_training(
    args,
    *,
    algo_name: str,
    model_name: str,
    build_agent: Callable,
    build_optimizer: Callable,
    update_fn: Callable,
    collect_rollout_fn: Callable,
    rollout_mode: str = "mappo",
    env_kwargs_list=None,
    train_mode_during_update: bool = False,
    init_label: str = "Online init",
    collate_fn_builder: Optional[Callable] = None,
    sample_fn_builder: Optional[Callable] = None,
):
    torch.backends.cudnn.deterministic = not args.ignore_torch_deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator(mixed_precision="bf16" if args.use_amp else "no")
    device = accelerator.device

    env_kwargs_list = DEFAULT_ENV_KWARGS_LIST if env_kwargs_list is None else env_kwargs_list
    ckpt = resolve_ckpt_dir(args, algo_name=algo_name, model_name=model_name)
    step_infos = get_step_infos(args)
    infos = get_agent_info(args)
    agent_names = infos["agent_names"]
    collate_fn = make_collate_fn(agent_names) if collate_fn_builder is None else collate_fn_builder(agent_names, device)

    agent = build_agent(args, infos, device=device) if "device" in inspect.signature(build_agent).parameters else build_agent(args, infos)
    optimizer = build_optimizer(args, agent)
    clients = None
    if getattr(args, "not_train_aggregator", False):
        ensure_feature_aggregator_args(args)
        clients = build_clients(args, agent, infos, agent_names, device)
        if not args.evaluate_mode:
            if args.resume_dir is not None:
                ckpt_prefix = "best_ag_" if args.resume_use_best_agent else "latest_ag_"
                initialize_client_graph(clients, agent, load_dir=ckpt["root_dir"], ckpt_prefix=ckpt_prefix)
            elif args.init_agent_path:
                initialize_client_graph(
                    clients,
                    agent,
                    load_dir=os.path.dirname(args.init_agent_path),
                    ckpt_prefix=infer_aggregator_ckpt_prefix(args.init_agent_path),
                )
            else:
                initialize_client_graph(clients, agent)
        set_client_feature_aggregator_requires_grad(clients, False)

    resume_info = maybe_load_initial_agent(args, ckpt, agent, init_label=init_label)
    global_steps = 0
    if os.path.exists(ckpt["latest_opt"]):
        latest_opt = torch.load(ckpt["latest_opt"], map_location="cpu")
        if resume_info["resumed"] and resume_info["used_best_agent"]:
            inferred_best_step = resume_info["best_step"]
            if inferred_best_step is not None:
                global_steps = int(inferred_best_step)
            else:
                global_steps = int(latest_opt["step"])
                print(
                    f"[Resume] best-agent step could not be inferred; "
                    f"fall back to latest optimizer step={global_steps}"
                )
        else:
            optimizer.load_state_dict(latest_opt["opt"])
            global_steps = int(latest_opt["step"])

    agent, optimizer = accelerator.prepare(agent, optimizer)

    if accelerator.is_main_process:
        writer = SummaryWriter(ckpt["log_dir"], purge_step=global_steps)
        print(f"[TensorBoard] Logging to {ckpt['log_dir']}")
    else:
        writer = None

    base_env_kwargs = build_base_env_kwargs(args)
    continual_env_schedule = build_continual_env_schedule(args, env_kwargs_list)
    current_env_index = 0
    training_start_time = time.monotonic()
    envs, eval_envs, current_env_name = make_envs_for_env_kwargs(
        args, ckpt, current_env_index, base_env_kwargs, env_kwargs_list
    )
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    def maybe_switch_envs():
        nonlocal envs, eval_envs, next_obs, next_done, current_env_index, current_env_name
        if continual_env_schedule is None:
            return False, False, None
        elapsed_minutes = (time.monotonic() - training_start_time) / 60.0
        scheduled_env_index = bisect.bisect_right(
            continual_env_schedule.change_time_points,
            elapsed_minutes,
        )
        if scheduled_env_index >= len(continual_env_schedule.env_kwarg_list):
            return False, True, elapsed_minutes
        if scheduled_env_index == current_env_index:
            return False, False, elapsed_minutes

        previous_env_name = current_env_name
        current_env_index = scheduled_env_index
        envs.close()
        eval_envs.close()
        envs, eval_envs, current_env_name = make_envs_for_env_kwargs(
            args, ckpt, current_env_index, base_env_kwargs, env_kwargs_list
        )
        next_obs, _ = envs.reset(seed=args.seed)
        next_done = torch.zeros(args.num_envs, device=device)
        print(
            f"[OnlineRL] switch env from {previous_env_name} to {current_env_name} "
            f"at elapsed={elapsed_minutes:.2f} minutes"
        )
        return True, False, elapsed_minutes

    if args.evaluate_mode:
        if args.eval_agent_dir is None:
            raise ValueError("Please provide --eval-agent-dir for evaluation mode")
        if clients is not None:
            initialize_client_graph(clients, accelerator.unwrap_model(agent), load_dir=args.eval_agent_dir, ckpt_prefix="best_ag_")
            set_client_feature_aggregator_requires_grad(clients, False)
            for client in clients.values():
                client.use_eval_feature_selector_strategy()
        unwrapped_agent = accelerator.unwrap_model(agent)
        unwrapped_agent.load_checkpoint_state_dict(
            torch.load(os.path.join(args.eval_agent_dir, "best_agent.pt"), map_location="cpu")
        )
        unwrapped_agent.eval()
        sample_fn = make_sample_fn(agent_names, unwrapped_agent, deterministic=True) if sample_fn_builder is None else sample_fn_builder(agent_names, unwrapped_agent, device, deterministic=True)
        eval_metrics = evaluate(
            n=args.eval_episodes,
            sample_fn=sample_fn,
            eval_envs=eval_envs,
        )
        payload = {k: float(v.mean()) for k, v in eval_metrics.items()}
        eval_envs.close()
        envs.close()
        if accelerator.is_main_process:
            dump_json(os.path.join(ckpt["root_dir"], "eval_metrics.json"), payload)
        return

    metrics_log = load_json(ckpt["metrics"]) if os.path.exists(ckpt["metrics"]) else []
    _, inferred_best_score = infer_best_step_from_metrics(ckpt["metrics"])
    best_score = -1.0 if inferred_best_score is None else float(inferred_best_score)
    start_time = time.time()
    last_save_skip = args.resume_dir is not None

    if args.minibatch_size == 0:
        args.minibatch_size = step_infos["rollot_steps"] // args.num_minibatch // args.grad_accum_steps

    pbar = tqdm(total=step_infos["total_steps"], initial=global_steps, ascii=True)
    agent.freeze_state_stats()

    while global_steps < step_infos["total_steps"]:
        switched, schedule_finished, elapsed_minutes = maybe_switch_envs()
        if writer is not None and elapsed_minutes is not None:
            writer.add_scalar("time/elapsed_minutes", elapsed_minutes, global_steps)
            writer.add_scalar("continual/current_env_index", current_env_index, global_steps)
        if schedule_finished:
            print(f"[OnlineRL] continual schedule finished at elapsed={elapsed_minutes:.2f} minutes")
            break
        if args.max_time is not None:
            elapsed_minutes = (time.monotonic() - training_start_time) / 60.0
            if elapsed_minutes >= args.max_time:
                print(f"[OnlineRL] reached max_time={args.max_time} minutes")
                break
        if switched:
            print(f"[OnlineRL] active env={current_env_name}")

        if clients is not None:
            for client in clients.values():
                client.train()
                client.use_train_feature_selector_strategy()

        unwrapped_agent = accelerator.unwrap_model(agent)
        unwrapped_agent.eval()

        if not last_save_skip and global_steps % step_infos["save_interval_steps"] == 0 and accelerator.is_main_process:
            save_checkpoint(ckpt["latest_agent"], unwrapped_agent.checkpoint_state_dict())
            save_checkpoint(ckpt["latest_opt"], {"opt": optimizer.state_dict(), "step": global_steps})
            if clients is not None:
                for name in agent_names:
                    clients[name].save_feature_aggregators(os.path.join(ckpt["root_dir"], f"latest_ag_{name}.pt"))
                set_client_feature_aggregator_requires_grad(clients, False)
                for client in clients.values():
                    client.use_eval_feature_selector_strategy()

            sample_fn = make_sample_fn(agent_names, unwrapped_agent, deterministic=True) if sample_fn_builder is None else sample_fn_builder(agent_names, unwrapped_agent, device, deterministic=True)
            eval_metrics = evaluate(
                n=args.eval_episodes,
                sample_fn=sample_fn,
                eval_envs=eval_envs,
            )
            if clients is not None:
                for client in clients.values():
                    client.use_train_feature_selector_strategy()
            eval_payload = {key: float(value.mean()) for key, value in eval_metrics.items()}
            for key, value in eval_payload.items():
                writer.add_scalar(f"eval/{key}", value, global_steps)
                print(f"eval_{key}_mean={value}")

            score = eval_payload.get("success_once", eval_payload.get("success_rate", next(iter(eval_payload.values()))))
            pbar.set_postfix(eval_score=score, env=current_env_name)
            metrics_log.append({"step": global_steps, "score": float(score), "env": current_env_name, "elapsed_minutes": float(elapsed_minutes)})
            dump_json(ckpt["metrics"], metrics_log)
            if score >= best_score:
                best_score = float(score)
                save_checkpoint(ckpt["best_agent"], unwrapped_agent.checkpoint_state_dict())
                if clients is not None:
                    for name in agent_names:
                        clients[name].save_feature_aggregators(os.path.join(ckpt["root_dir"], f"best_ag_{name}.pt"))
                print(f"[Eval] New best model saved (score={score:.3f}, env={current_env_name})")

        last_save_skip = False

        rollout_kwargs = dict(
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
        if "clients" in inspect.signature(collect_rollout_fn).parameters:
            rollout_kwargs["clients"] = clients
        rollout = collect_rollout_fn(**rollout_kwargs)

        if rollout_mode == "maple":
            (
                obs_buf,
                action_buf,
                logp_buf,
                rew_buf,
                done_buf,
                val_buf,
                final_val_buf,
                next_obs,
                next_done,
                meta,
            ) = rollout
        else:
            if len(rollout) == 10:
                (
                    obs_buf,
                    action_buf,
                    logp_buf,
                    rew_buf,
                    done_buf,
                    val_buf,
                    final_val_buf,
                    next_obs,
                    next_done,
                    old_token_logits_buf,
                ) = rollout
            else:
                (
                    obs_buf,
                    action_buf,
                    logp_buf,
                    rew_buf,
                    done_buf,
                    val_buf,
                    final_val_buf,
                    next_obs,
                    next_done,
                ) = rollout
                old_token_logits_buf = None

        adv_buf, ret_buf = compute_gae(
            rew_buf,
            done_buf,
            val_buf,
            final_val_buf,
            next_obs,
            next_done,
            agent,
            collate_fn,
            args,
            accelerator,
        )

        if rollout_mode == "maple":
            data = (
                obs_buf.reshape((-1,)),
                action_buf.reshape((-1,)),
                logp_buf.reshape((-1,)),
                adv_buf.reshape(-1),
                ret_buf.reshape(-1),
                val_buf.reshape(-1),
                flatten_meta(meta),
            )
        else:
            data = (
                obs_buf.reshape((-1,)),
                action_buf.reshape((-1,)),
                logp_buf.reshape((-1,)),
                adv_buf.reshape(-1),
                ret_buf.reshape(-1),
                val_buf.reshape(-1),
            )
            if old_token_logits_buf is not None:
                data = data + (old_token_logits_buf.reshape((-1,)),)

        if train_mode_during_update:
            agent.train()
        else:
            agent.eval()
        stats = update_fn(
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

        if clients is not None:
            ag_data_infos = {}
            for client_name, client in clients.items():
                ag_data_infos[client_name] = client.export_feature_and_action()
            if accelerator.is_main_process and writer is not None:
                log_uploaded_q_task_stats(
                    writer,
                    ag_data_infos,
                    global_steps,
                    enable_histograms=not getattr(args, "disable_ag_debug_histograms", False),
                )
                log_effective_transmission_stats(
                    writer,
                    clients,
                    global_steps,
                )
                log_agent_feature_cosine_similarity(writer, ag_data_infos, global_steps)
            for recv_name, client_recv in clients.items():
                for send_name, ag_data in ag_data_infos.items():
                    if recv_name != send_name:
                        client_recv.receive_feature_and_action(send_name, ag_data)
            if accelerator.is_main_process and writer is not None:
                for client_name, client in clients.items():
                    feature_aggregators_gate_g = client.debug_feature_aggregators()
                    gate_g_strs = []
                    for remote_name, gate_info in feature_aggregators_gate_g.items():
                        for stream_name, gate_g in gate_info.items():
                            if gate_g is None:
                                continue
                            metric_prefix = (
                                f"monitoring/aggregation/gate_g/{client_name}/from_{remote_name}/{stream_name}"
                            )
                            if not getattr(args, "disable_ag_debug_histograms", False):
                                writer.add_histogram(metric_prefix, gate_g, global_steps)
                            writer.add_scalar(f"{metric_prefix}_mean", gate_g.mean(), global_steps)
                            writer.add_scalar(f"{metric_prefix}_std", gate_g.std(), global_steps)
                            gate_g_strs.append(f"{remote_name}.{stream_name}={gate_g.mean().item():.4f}")
                    if gate_g_strs:
                        print(f'Client {client_name} gate_g_mean: {", ".join(gate_g_strs)}')
                log_feature_aggregator_agent_cosine_similarity(writer, clients, global_steps)

        agent.unfreeze_state_stats()
        agent.update_state_stats(obs_buf.reshape((-1,)), update_actor=False, update_critic=True)
        agent.update_state_stats(next_obs, update_actor=False, update_critic=True)
        agent.freeze_state_stats()

        global_steps += step_infos["rollot_steps"]
        pbar.update(step_infos["rollot_steps"])

        if accelerator.is_main_process and writer is not None:
            y_pred = val_buf.flatten(0, 1).cpu().numpy()
            y_true = ret_buf.flatten(0, 1).cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            sps = global_steps / max(time.time() - start_time, 1e-6)
            writer.add_scalar("charts/SPS", sps, global_steps)
            writer.add_scalar("loss/policy", stats.get("policy_loss", 0.0), global_steps)
            writer.add_scalar("loss/value", stats.get("value_loss", 0.0), global_steps)
            writer.add_scalar("loss/entropy", stats.get("entropy", 0.0), global_steps)
            writer.add_scalar("loss/approx_kl", stats.get("approx_kl", 0.0), global_steps)
            writer.add_scalar("loss/old_approx_kl", stats.get("old_approx_kl", 0.0), global_steps)
            writer.add_scalar("loss/clip_frac", stats.get("clip_frac", 0.0), global_steps)
            writer.add_scalar("loss/full_kl", stats.get("full_kl", 0.0), global_steps)
            writer.add_scalar("loss/argmax_change_frac", stats.get("argmax_change_frac", 0.0), global_steps)
            writer.add_scalar("loss/future_state", stats.get("future_state_loss", 0.0), global_steps)
            writer.add_scalar("loss/bc", stats.get("bc_loss", 0.0), global_steps)
            writer.add_scalar("loss/explained_var", explained_var, global_steps)

    envs.close()
    eval_envs.close()
    if accelerator.is_main_process and writer is not None:
        unwrapped_agent = accelerator.unwrap_model(agent)
        save_checkpoint(ckpt["latest_agent"], unwrapped_agent.checkpoint_state_dict())
        save_checkpoint(ckpt["latest_opt"], {"opt": optimizer.state_dict(), "step": global_steps})
        if clients is not None:
            for name in agent_names:
                clients[name].save_feature_aggregators(os.path.join(ckpt["root_dir"], f"latest_ag_{name}.pt"))
        writer.close()
