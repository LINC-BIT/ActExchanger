import os
import sys
import time
from datetime import datetime

os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))

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
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import envs.two_robot_pick_cube_v2  # noqa: F401
from mani_skill.utils.io_utils import dump_json, load_json
from train.internVL.checkpoint_utils import load_agent_checkpoint
from train.internVL.model import build_batch_from_obs
from train.marl.maple import MapleEdgeVLAAgent, build_optimizer, collect_rollout, maple_update_on_policy
from train.reinforcement_learning.evaluate import evaluate
from train.reinforcement_learning.make_env import make_eval_envs
from train.reinforcement_learning.utils import compute_gae, get_stage, get_step_infos

mp.set_start_method("spawn", force=True)


MODEL_NAME = "internvl3_5_1b_instruct_maple_rl"


def resolve_ckpt_dir(args):
    task_dir = os.path.join(args.save_dir, f"{args.task_name}/maple/{args.robot_name}/{MODEL_NAME}")
    root_dir = os.path.join(task_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(task_dir, exist_ok=True)

    resume_dir = args.resume_dir if args.resume_dir is not None else root_dir
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
        "global_state_dim": batch["global_state"].shape[-1],
        "action_dim": test_env.single_action_space[agent_names[0]].shape[0],
    }
    test_env.close()
    return info


def save_checkpoint(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


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


def set_optimizer_phase_lrs(args, optimizer, critic_only: bool) -> None:
    actor_group_names = {
        "vla",
        "state_projector",
        "context_projector",
        "actor_head",
        "vla_adapter_backbone",
        "smolvla_backbone",
        "efficientvla_backbone",
        "vla_adapter_state_projector",
        "smolvla_state_projector",
        "efficientvla_state_projector",
        "vla_adapter_heads",
        "smolvla_heads",
        "efficientvla_heads",
        "latent_coordination",
    }
    critic_group_names = {
        "critic",
        "critic_state_encoder",
        "critic_visual_encoder",
    }
    future_group_names = {"future_state_head"}
    for group in optimizer.param_groups:
        group_name = group.get("group_name")
        if group_name in critic_group_names:
            group["lr"] = args.value_head_learning_rate
        elif group_name in future_group_names:
            group["lr"] = 0.0 if critic_only else getattr(args, "future_head_learning_rate", args.head_learning_rate)
        elif group_name in actor_group_names:
            group["lr"] = 0.0 if critic_only else group.get("initial_lr", group["lr"])


def update_state_stats_critic_only(agent, obs) -> None:
    try:
        agent.update_state_stats(obs, update_actor=False, update_critic=True)
    except TypeError:
        agent.update_state_stats(obs)


def main(args):
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

    agent = MapleEdgeVLAAgent(
        **infos,
        model_dir=args.model_dir,
        normalize_state=args.normalize_state,
        freeze_vla_backbone=args.freeze_vla_backbone,
        critic_hidden_dim=args.critic_hidden_dim,
        attention_implementation=args.attn_implementation,
        image_size=args.image_size,
        use_vla_lora=args.use_vla_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        latent_layers=args.maple_latent_layers,
        future_horizon=args.maple_future_horizon,
        use_vision_lora=getattr(args, "use_vision_lora", False),
        train_vision_backbone=getattr(args, "train_vision_backbone", False),
        vision_token_pool_size=getattr(args, "vision_token_pool_size", None),
        policy_mode=getattr(args, "policy_mode", "native"),
        tiny_hidden_dim=getattr(args, "tiny_hidden_dim", 384),
        tiny_vision_layers=getattr(args, "tiny_vision_layers", 4),
        tiny_decoder_layers=getattr(args, "tiny_decoder_layers", 4),
        tiny_attention_heads=getattr(args, "tiny_attention_heads", 6),
        tiny_patch_size=getattr(args, "tiny_patch_size", 14),
        tiny_ffn_mult=getattr(args, "tiny_ffn_mult", 4),
        tiny_num_action_bins=getattr(args, "tiny_num_action_bins", 256),
        tiny_prompt_length=getattr(args, "tiny_prompt_length", 24),
        smolvla_hidden_dim=getattr(args, "smolvla_hidden_dim", 384),
        smolvla_vision_layers=getattr(args, "smolvla_vision_layers", 6),
        smolvla_attention_heads=getattr(args, "smolvla_attention_heads", 6),
        smolvla_patch_size=getattr(args, "smolvla_patch_size", 14),
        smolvla_ffn_mult=getattr(args, "smolvla_ffn_mult", 4),
        efficientvla_hidden_dim=getattr(args, "efficientvla_hidden_dim", 256),
        efficientvla_vision_layers=getattr(args, "efficientvla_vision_layers", 4),
        efficientvla_decoder_layers=getattr(args, "efficientvla_decoder_layers", 1),
        efficientvla_attention_heads=getattr(args, "efficientvla_attention_heads", 4),
        efficientvla_patch_size=getattr(args, "efficientvla_patch_size", 16),
        efficientvla_ffn_mult=getattr(args, "efficientvla_ffn_mult", 3),
    )
    optimizer = build_optimizer(args, agent)

    if os.path.exists(ckpt["latest_agent"]):
        agent.load_checkpoint_state_dict(torch.load(ckpt["latest_agent"], map_location="cpu"))
    elif args.init_agent_path:
        load_agent_checkpoint(agent, args.init_agent_path, map_location="cpu", label="MAPLE online RL init")
    global_steps = 0
    if os.path.exists(ckpt["latest_opt"]):
        latest_opt = torch.load(ckpt["latest_opt"], map_location="cpu")
        optimizer.load_state_dict(latest_opt["opt"])
        global_steps = int(latest_opt["step"])

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
        print("[Evaluate] Start evaluation only mode")
        eval_metrics = evaluate(
            n=args.eval_episodes,
            sample_fn=make_sample_fn(agent_names, unwrapped_agent, deterministic=True),
            eval_envs=eval_envs,
        )
        payload = {k: float(v.mean()) for k, v in eval_metrics.items()}
        if accelerator.is_main_process:
            for key, value in eval_metrics.items():
                mean = value.mean()
                print(f"eval_{key}_mean={mean}")
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
    best_score = -1.0
    start_time = time.time()
    last_save_skip = args.resume_dir is not None

    if args.minibatch_size == 0:
        args.minibatch_size = step_infos["rollot_steps"] // args.num_minibatch // args.grad_accum_steps

    pbar = tqdm(total=step_infos["total_steps"], initial=global_steps, ascii=True)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    current_phase = None
    critic_only = global_steps < step_infos["critic_warmup_steps"]
    set_optimizer_phase_lrs(args, optimizer, critic_only=critic_only)
    agent.unfreeze_state_stats()
    update_state_stats_critic_only(agent, next_obs)
    agent.freeze_state_stats()

    while global_steps < step_infos["total_steps"]:
        critic_only = global_steps < step_infos["critic_warmup_steps"]
        phase = "critic_only" if critic_only else "full_ppo"
        if phase != current_phase:
            current_phase = phase
            print(
                f"[Phase] step={global_steps} mode={phase} "
                f"(critic_warmup_steps={step_infos['critic_warmup_steps']})"
            )
        args.critic_only_update = critic_only
        set_optimizer_phase_lrs(args, optimizer, critic_only=critic_only)

        unwrapped_agent = accelerator.unwrap_model(agent)
        unwrapped_agent.eval()

        if not last_save_skip and global_steps % step_infos["save_interval_steps"] == 0 and accelerator.is_main_process:
            save_checkpoint(ckpt["latest_agent"], unwrapped_agent.checkpoint_state_dict())
            save_checkpoint(ckpt["latest_opt"], {"opt": optimizer.state_dict(), "step": global_steps})

            eval_metrics = evaluate(
                n=args.eval_episodes,
                sample_fn=make_sample_fn(agent_names, unwrapped_agent, deterministic=True),
                eval_envs=eval_envs,
            )
            for key, value in eval_metrics.items():
                mean = value.mean()
                writer.add_scalar(f"eval/{key}", mean, global_steps)
                print(f"eval_{key}_mean={mean}")

            score = eval_metrics.get("success_rate", eval_metrics[list(eval_metrics.keys())[0]]).mean()
            pbar.set_postfix(eval_score=score)
            metrics_log.append({"step": global_steps, "score": float(score)})
            dump_json(ckpt["metrics"], metrics_log)
            if score >= best_score:
                best_score = score
                save_checkpoint(ckpt["best_agent"], unwrapped_agent.checkpoint_state_dict())
                print(f"[Eval] New best model saved (score={score:.3f})")

        last_save_skip = False

        rollout = collect_rollout(
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
        obs_buf, action_bin_buf, logp_buf, rew_buf, done_buf, val_buf, final_val_buf, next_obs, next_done, meta = rollout

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

        data = (
            obs_buf.reshape((-1,)),
            action_bin_buf.reshape((-1,)),
            logp_buf.reshape((-1,)),
            adv_buf.reshape(-1),
            ret_buf.reshape(-1),
            val_buf.reshape(-1),
            flatten_meta(meta),
        )

        agent.eval()
        stats = maple_update_on_policy(
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
        update_state_stats_critic_only(agent, obs_buf.reshape((-1,)))
        update_state_stats_critic_only(agent, next_obs)
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
            writer.add_scalar("loss/future_state", stats["future_state_loss"], global_steps)
            writer.add_scalar("loss/bc", stats["bc_loss"], global_steps)
            writer.add_scalar("loss/adv_std", stats["adv_std"], global_steps)
            writer.add_scalar("loss/explained_var", explained_var, global_steps)

    if accelerator.is_main_process:
        unwrapped_agent = accelerator.unwrap_model(agent)
        save_checkpoint(ckpt["latest_agent"], unwrapped_agent.checkpoint_state_dict())
        save_checkpoint(ckpt["latest_opt"], {"opt": optimizer.state_dict(), "step": global_steps})
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str, default="TwoRobotPickCube-v2")
    parser.add_argument("--seed", type=int, default=1788)
    parser.add_argument("--total-steps", type=int, default=20_000_000)
    parser.add_argument("--critic-warmup-rollouts", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-eval-envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--ignore-partial-reset", action="store_true")
    parser.add_argument("--ignore-torch-deterministic", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--num-minibatch", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--normalize-state", action="store_true")
    parser.add_argument("--not-normalize-adv", action="store_true")
    parser.add_argument("--finite-horizon-gae", action="store_true")
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-5)
    parser.add_argument("--head-learning-rate", type=float, default=3e-5)
    parser.add_argument("--state-learning-rate", type=float, default=3e-5)
    parser.add_argument("--value-head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--future-head-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.956)
    parser.add_argument("--gae-lambda", type=float, default=0.966)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument(
        "--clip-vloss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use clipped Huber value loss for critic updates.",
    )
    parser.add_argument("--value-huber-delta", type=float, default=10.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.2)
    parser.add_argument("--save-dir", type=str, default="ckpt")
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--save-interval-per-rollout", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--evaluate-mode", action="store_true")
    parser.add_argument("--eval-agent-dir", type=str, default=None)
    parser.add_argument("--robot-name", type=str, default="pandas_pandas")
    parser.add_argument("--model-dir", type=str, default="OpenGVLab/InternVL3_5-1B-Instruct")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--init-agent-path", type=str, default=None)
    parser.add_argument("--obs-mode", type=str, default="rgb+state_dict")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pos")
    parser.add_argument("--reward-mode", type=str, default="normalized_dense")
    parser.add_argument("--freeze-vla-backbone", action="store_true")
    parser.add_argument("--use-vla-lora", action="store_true")
    parser.add_argument("--use-vision-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-vision-backbone", action="store_true")
    parser.add_argument("--model-backbone", type=str, default=None)
    parser.add_argument("--policy-mode", type=str, default="native")
    parser.add_argument("--vision-token-pool-size", type=int, default=None)
    parser.add_argument("--tiny-hidden-dim", type=int, default=384)
    parser.add_argument("--tiny-vision-layers", type=int, default=4)
    parser.add_argument("--tiny-decoder-layers", type=int, default=4)
    parser.add_argument("--tiny-attention-heads", type=int, default=6)
    parser.add_argument("--tiny-patch-size", type=int, default=14)
    parser.add_argument("--tiny-ffn-mult", type=int, default=4)
    parser.add_argument("--tiny-num-action-bins", type=int, default=256)
    parser.add_argument("--tiny-prompt-length", type=int, default=24)
    parser.add_argument("--smolvla-hidden-dim", type=int, default=384)
    parser.add_argument("--smolvla-vision-layers", type=int, default=6)
    parser.add_argument("--smolvla-attention-heads", type=int, default=6)
    parser.add_argument("--smolvla-patch-size", type=int, default=14)
    parser.add_argument("--smolvla-ffn-mult", type=int, default=4)
    parser.add_argument("--efficientvla-hidden-dim", type=int, default=256)
    parser.add_argument("--efficientvla-vision-layers", type=int, default=4)
    parser.add_argument("--efficientvla-decoder-layers", type=int, default=1)
    parser.add_argument("--efficientvla-attention-heads", type=int, default=4)
    parser.add_argument("--efficientvla-patch-size", type=int, default=16)
    parser.add_argument("--efficientvla-ffn-mult", type=int, default=3)
    parser.add_argument("--critic-hidden-dim", type=int, default=512)
    parser.add_argument("--maple-group-size", type=int, default=4)
    parser.add_argument("--maple-grpo-eps", type=float, default=1e-5)
    parser.add_argument("--maple-gamma", type=float, default=0.956)
    parser.add_argument("--maple-ent-coef", type=float, default=0.0)
    parser.add_argument("--maple-future-state-coef", type=float, default=0.1)
    parser.add_argument("--maple-future-state-loss", type=str, default="huber", choices=["huber", "mse", "l1"])
    parser.add_argument("--maple-future-state-delta", type=float, default=1.0)
    parser.add_argument("--maple-rl-bc-coef", type=float, default=0.0)
    parser.add_argument("--maple-latent-layers", type=int, default=2)
    parser.add_argument("--maple-future-horizon", type=int, default=1)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
