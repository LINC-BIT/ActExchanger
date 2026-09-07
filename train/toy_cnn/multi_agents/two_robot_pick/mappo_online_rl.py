# import threading
# import time
# import traceback
# import sys

# def monitor_threads(log_file="thread_log.txt", interval=2):
#     """周期性打印所有线程状态和堆栈"""
#     with open(log_file, "w") as f:
#         while True:
#             f.write("="*40 + "\n")
#             for t in threading.enumerate():
#                 f.write(f"Thread {t.name} (id={t.ident}): alive={t.is_alive()}\n")
#                 if t.ident in sys._current_frames():
#                     stack = ''.join(traceback.format_stack(sys._current_frames()[t.ident]))
#                     f.write(stack + "\n")
#             f.write("="*40 + "\n\n")
#             f.flush()  # 保证实时写入文件
#             time.sleep(interval)

# threading.Thread(target=monitor_threads, daemon=True, name="ThreadMonitor").start() 

import os
import sys
sys.path.append(os.getcwd())
from train.toy_cnn.multi_agents.two_robot_pick.gpu_auto_select import configure_cuda_visible_devices

configure_cuda_visible_devices()
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
from datetime import datetime
import argparse
import itertools
import time
from tqdm import tqdm
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import torch
import torch.nn as nn
print(torch.cuda.device_count())
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

from accelerate import Accelerator
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from mani_skill.utils.io_utils import load_json, dump_json
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import gymnasium as gym

import random
import bisect

from train.toy_cnn.model import MAPPOAgent
from train.reinforcement_learning.utils import compute_gae, get_step_infos, get_stage
from train.marl.mappo.base import collect_rollout, mappo_update_on_policy, mappo_update_on_policy_ag
from train.reinforcement_learning.make_env import make_eval_envs
from train.reinforcement_learning.evaluate import evaluate
import envs.two_robot_pick_cube_v2
import copy
from copy import deepcopy
from PIL import Image
from ours.utils.dl.common.model import get_module, set_module
from ours.libs.train_with_fbs.lib_transformer import svd_decompose_linear
from ours.pretrain_fbs_model.main import add_FBS_into_cnn
from ours.de_feature_fusion.client import ClientForMultiAgent
from ours.libs.train_with_fbs.lib import set_sparsity
from online_utils import build_continual_env_schedule

# =====================================================
# Preprocess
# =====================================================
model_name = 'toy_cnn'
robot_name = 'pandas_pandas'
sensor_configs = {
    "shader_pack": "default",
    "width": 128,
    "height": 128
}
cameras=("hand_camera",)

# env_kwargs_list = [
#     {
#         "name" : 'ObjectScaleUp1p4',
#         "object_scale": 1.4,
#     },
#     {
#         "name" : 'LightStronger80',
#         "light_intensity": 1.8,
#     },
#     {
#         "name" : 'LightWeaker80',
#         "light_intensity": 0.5,
#     },
#     {
#         "name" : 'TempHigher70',
#         "ambient_light_temperature": 1.7,
#     },
#     {
#         "name" : 'TempLower70',
#         "ambient_light_temperature": 0.3,
#     },
#     {
#         "name" : 'ObjectScaleDown0p6',
#         "object_scale": 0.6,
#     },
#     {
#         "name" : 'MassScaleUp1p4',
#         "mass_scale": 1.4,
#     },
#     {
#         "name" : 'MassScaleDown0p6',
#         "mass_scale": 0.6,
#     },
#     {
#         "name" : 'ObjectColorChangeGreen',
#         "object_color": "green",
#     },
#     {
#         "name" : 'ObjectColorChangeBlue',
#         "object_color": "blue",
#     },
# ]


# env_kwargs_list = [
#     {
#         "name" : 'CameraRotate+155',
#         "camera_y_rotate": 155,
#         "object_scale": 0.5,
#     },
#     {
#         "name" : 'ObjectScaleDown0p5',
#         "object_scale": 0.5,
#     },
#     {
#         "name" : 'ChangeShapeToSphere',
#         "object_type": "sphere",
#         "object_scale": 0.5,
#     },
#     {
#         "name" : 'CameraRotate+25',
#         "camera_y_rotate": 25,
#         "object_scale": 0.5,
#     },
#     {
#         "name" : 'ChangeShapeToCylinder',
#         "object_type": "cylinder",
#         "object_scale": 0.5,
#     },
     
#     {
#         "name" : 'CameraRotate+155',
#         "camera_y_rotate": 155,
#         "object_scale": 0.5,
#     },
#     {
#         "name" : 'ObjectScaleDown0p5',
#         "object_scale": 0.5,
#     },
#     {
#         "name" : 'ChangeShapeToSphere',
#         "object_type": "sphere",
#         "object_scale": 0.5,
#     },
#     {
#         "name" : 'CameraRotate+25',
#         "camera_y_rotate": 25,
#         "object_scale": 0.5,
#     },
#     {
#         "name" : 'ChangeShapeToCylinder',
#         "object_type": "cylinder",
#         "object_scale": 0.5,
#     },
# ]

env_kwargs_list = [
    {
        "name" : 'ChangeShapeToSphere0p6',
        "object_type": "sphere",
        "object_scale": 0.6,
    },
    {
        "name" : 'ChangeShapeToCube0p6',
        "object_scale": 0.6,
    },
    {
        "name" : 'ChangeShapeToCylinder0p6',
        "object_type": "cylinder",
        "object_scale": 0.6,
    },
    {
        "name" : 'ChangeShapeToSphere0p5',
        "object_type": "sphere",
        "object_scale": 0.5,
    },
    {
        "name" : 'ChangeShapeToCylinder0p5',
        "object_type": "cylinder",
        "object_scale": 0.5,
    },
    {
        "name" : 'ChangeShapeToSphere0p6',
        "object_type": "sphere",
        "object_scale": 0.6,
    },
    {
        "name" : 'ChangeShapeToCube0p6',
        "object_scale": 0.6,
    },
    {
        "name" : 'ChangeShapeToCylinder0p6',
        "object_type": "cylinder",
        "object_scale": 0.6,
    },
    {
        "name" : 'ChangeShapeToSphere0p5',
        "object_type": "sphere",
        "object_scale": 0.5,
    },
    {
        "name" : 'ChangeShapeToCylinder0p5',
        "object_type": "cylinder",
        "object_scale": 0.5,
    },
]

def process_agent(args, infos, env_kwargs, agent_obs_rules, device):
    agent = MAPPOAgent(**infos, normalize_state=args.normalize_state, use_depth=False).to(device)
    
    # agent.reset_logstd()
    # agent.reset_value_head()
    
    # add FBS
    set_module(agent, 'rgb_encoder.fc.0', svd_decompose_linear(
        get_module(agent, 'rgb_encoder.fc.0')
    ))
    dummy_env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend='gpu' if device.type == 'cuda' else 'cpu',
        env_kwargs=env_kwargs,
        wrappers=[lambda env: FlattenRGBObservationWrapperForMARL(env, agent_obs_rules=agent_obs_rules)],
    )
    example_sample, _ = dummy_env.reset(seed=args.seed)
    action_heads_names = [f'actor_heads.{name}.0' for name in agent.actor_heads.keys()]
    collate_fn = make_collate_fn(args, cameras, device)
    example_sample = collate_fn(example_sample)
    # Only extract action feature
    add_FBS_into_cnn(
        agent,
        [f'rgb_encoder.cnn.{i}' for i in [0, 6, 12]],
        action_heads_names + ['rgb_encoder.fc.0.0', 'critic.0'],
        # ['rgb_encoder.fc.0.0', 'critic.0'],
        example_sample,
        args.max_sparsity,
        8,
        lambda model, sample: model(sample)
    )
    agent.load_state_dict(torch.load(os.path.join(args.pretrained_agent_path, 'best_agent.pt'), map_location=device))
    print(f'[load] load pretrained fbs model from {args.pretrained_agent_path}')

    for m in agent.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    print('agent: ', agent)

    return agent

class FlattenRGBObservationWrapperForMARL(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation

    Note that the returned observations will have a "rgb" key depending on the rgb bool flags, and will
    always have a "state" key. 
    """

    def __init__(self, env, agent_obs_rules, rgb=True, state=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_state = state

        # check if rgb data exists in first camera's sensor data
        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "rgb" not in first_cam:
            self.include_rgb = False
        self.agent_obs_rules = agent_obs_rules # a dict of agent_name: list of keys to include in the observation for that agent
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        rgb_images = {}

        for k, cam_data in sensor_data.items():
            if self.include_rgb:
                rgb_images[k] = cam_data["rgb"]

        # flatten the rest of the data which should just be state data
        # print(observation)
        # 若有base_camera
        Image.fromarray(rgb_images['base_camera'][0].cpu().numpy()).save("debug_rgb.png")

        agent_states = {}
        for agent_name in observation['agent'].keys():
            observation_states = {}
            for key in self.agent_obs_rules[agent_name]:
                if key in observation['agent'][agent_name]:
                    observation_states[key] = observation['agent'][agent_name][key].to(self.base_env.device)
                elif key in observation['extra']:
                    observation_states[key] = observation['extra'][key].to(self.base_env.device)
                else:
                    raise ValueError(f"Key {key} not found in agent obs or extra obs")

            observation_states = common.flatten_state_dict(
                observation_states, use_torch=True, device=self.base_env.device
            )
            
            agent_states[agent_name] = observation_states

        global_state = common.flatten_state_dict(
            observation, use_torch=True, device=self.base_env.device
        )

        ret = dict()
        if self.include_rgb:
            assert len(rgb_images) == 1, "Currently only support one camera view for each agent, please specify the camera view you want in the agent_obs_rules and make sure only that camera view is included in the sensor data"
            ret['rgb'] = rgb_images["base_camera"]
        if self.include_state:
            for name, state in agent_states.items():
                ret[f'agent_states_{name}'] = state
            ret['global_state'] = global_state

        del observation
        # observation['goal_pos'] = observation['extra']['goal_pos'].to(self.base_env.device)
        # observation['tcp_pose'] = observation['extra']['tcp_pose'].to(self.base_env.device)
        # observation['tcp_to_goal_pos'] = observation['extra']['tcp_to_goal_pos'].to(self.base_env.device)
        # observation['agent']['qpos'] = observation['agent']['qpos'][:, :6]
        # observation['agent']['qvel'] = observation['agent']['qvel'][:, :6]
        # for j in self.base_env.agent.robot.get_active_joints():
        #     print(j.name)
        # print(self.base_env.agent.robot.get_qpos())

        return ret

def resolve_ckpt_dir(args, model_name):
    
    task_dir = os.path.join(args.save_dir, f'{args.task_name}_online{"_wo_ag" if args.not_train_aggregator else ""}{"_baseline" if args.baseline else ""}/ppo/{args.robot_name}/{model_name}')
    root_dir = os.path.join(task_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    os.makedirs(task_dir, exist_ok=True)

    if args.resume_dir is not None:
        resume_dir = args.resume_dir
    else:
        resume_dir = root_dir

    log_dir = os.path.join(resume_dir, "tb")
    video_dir=os.path.join(resume_dir, 'videos')
    latest_agent = os.path.join(resume_dir, "latest_agent.pt")
    latest_opt = os.path.join(resume_dir, "latest_opt.pt")
    actor_path = os.path.join(args.actor_ckpt_path if args.actor_ckpt_path is not None else '', "last.pt")

    return {
        "task_dir": task_dir,
        "log_dir": log_dir,
        "root_dir": root_dir,
        "video_dir": video_dir,
        "latest_agent": latest_agent,
        "latest_opt": latest_opt,
        "best_agent": os.path.join(resume_dir, "best_agent.pt"),
        "metrics": os.path.join(resume_dir, "metrics.json"),
        "actor_path": actor_path,
        "ft_agent_path": os.path.join(args.ft_agent_path if args.ft_agent_path is not None else '', "best_agent.pt"),
    }

def make_collate_fn(args, cameras, device):
    # device 指将obs移动到什么device, 不是obs是什么device
    def _resize(img, size=128):
        # img = img.unsqueeze(0)          # (1, C, H, W)
        img = F.interpolate(
            img,
            size=size,
            mode='bilinear',
            # align_corners=align_corners if mode != "nearest" else None,
        )
        return img
    
    def collate_fn(obs):
        if isinstance(obs['rgb'], np.ndarray):
            rgb = torch.from_numpy(obs['rgb'] / 255.0).permute(0, 3, 1, 2).float()
            global_state = torch.from_numpy(obs['global_state'])
            agent_states = {}
            for k, state in obs.items():
                if k not in ['rgb', 'global_state']:
                    agent_states[k] = torch.from_numpy(state)
            
        else:
            rgb = obs["rgb"].permute(0, 3, 1, 2).float() / 255.0  # (env, H, W, 3)
            global_state = obs['global_state']
            agent_states = {}
            for k, state in obs.items():
                if k not in ['rgb', 'global_state']:
                    agent_states[k] = state

        rgb = _resize(rgb).to(device)
        for k in agent_states:
            agent_states[k] = agent_states[k].to(device)
        global_state = global_state.to(device)

        ret = {
            'rgb': rgb,
            'global_state': global_state,
        }
        ret.update(agent_states)

        return ret

    return collate_fn


def _summarize_uploaded_feature(feature_message):
    if feature_message is None or feature_message.get("feature") is None:
        return None

    feature = feature_message["feature"].detach().float()
    if feature.ndim == 2:
        feature = feature.unsqueeze(0)
    if feature.ndim != 3:
        return None

    valid_mask = feature.abs().sum(dim=-1) > 0
    if not valid_mask.any():
        return None

    return feature[valid_mask].mean(dim=0)


def log_agent_feature_cosine_similarity(writer, ag_data_infos, global_steps, metric_prefix="monitoring/communication/agent_feature_cosine"):
    feature_summaries = {}
    for agent_name, feature_message in ag_data_infos.items():
        summary = _summarize_uploaded_feature(feature_message)
        if summary is not None:
            feature_summaries[agent_name] = summary

    cosine_values = []
    for agent_a, agent_b in itertools.combinations(sorted(feature_summaries.keys()), 2):
        cos_sim = F.cosine_similarity(
            feature_summaries[agent_a].unsqueeze(0),
            feature_summaries[agent_b].unsqueeze(0),
            dim=1,
            eps=1e-8,
        ).item()
        writer.add_scalar(f"{metric_prefix}/{agent_a}_vs_{agent_b}", cos_sim, global_steps)
        cosine_values.append(cos_sim)

    if cosine_values:
        cosine_tensor = torch.tensor(cosine_values, dtype=torch.float32)
        writer.add_scalar(f"{metric_prefix}_mean", cosine_tensor.mean().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}_min", cosine_tensor.min().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}_max", cosine_tensor.max().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}_std", cosine_tensor.std(unbiased=False).item(), global_steps)


def log_uploaded_q_task_stats(writer, ag_data_infos, global_steps, metric_prefix="monitoring/communication/uploaded_q_task"):
    all_q_task_values = []
    q_task_debug_strs = []

    for agent_name, feature_message in ag_data_infos.items():
        if feature_message is None:
            continue

        meta = feature_message.get("meta")
        if not isinstance(meta, dict):
            continue

        q_task = meta.get("q_task")
        if q_task is None:
            continue

        if not torch.is_tensor(q_task):
            q_task = torch.as_tensor(q_task, dtype=torch.float32)
        q_task = q_task.detach().float().view(-1).cpu()
        if q_task.numel() == 0:
            continue

        writer.add_histogram(f"{metric_prefix}/{agent_name}", q_task, global_steps)
        writer.add_scalar(f"{metric_prefix}/{agent_name}_mean", q_task.mean().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}/{agent_name}_min", q_task.min().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}/{agent_name}_max", q_task.max().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}/{agent_name}_std", q_task.std(unbiased=False).item(), global_steps)

        q_task_values = ", ".join(f"{value:.3f}" for value in q_task.tolist())
        q_task_debug_strs.append(
            f"{agent_name}: mean={q_task.mean().item():.3f}, min={q_task.min().item():.3f}, "
            f"max={q_task.max().item():.3f}, values=[{q_task_values}]"
        )
        all_q_task_values.append(q_task)

    if all_q_task_values:
        all_q_task = torch.cat(all_q_task_values, dim=0)
        writer.add_scalar(f"{metric_prefix}_mean", all_q_task.mean().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}_min", all_q_task.min().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}_max", all_q_task.max().item(), global_steps)
        writer.add_scalar(f"{metric_prefix}_std", all_q_task.std(unbiased=False).item(), global_steps)

    if q_task_debug_strs:
        print(f'Uploaded q_task: {" | ".join(q_task_debug_strs)}')


def _mean_summary_tensors(summary_tensors):
    valid_tensors = [tensor.float() for tensor in summary_tensors if tensor is not None]
    if not valid_tensors:
        return None
    return torch.stack(valid_tensors, dim=0).mean(dim=0)


def _log_pairwise_cosine_stats(writer, feature_by_agent, global_steps, metric_prefix):
    cosine_values = []
    for agent_a, agent_b in itertools.combinations(sorted(feature_by_agent.keys()), 2):
        cos_sim = F.cosine_similarity(
            feature_by_agent[agent_a].unsqueeze(0),
            feature_by_agent[agent_b].unsqueeze(0),
            dim=1,
            eps=1e-8,
        ).item()
        writer.add_scalar(f"{metric_prefix}/{agent_a}_vs_{agent_b}", cos_sim, global_steps)
        cosine_values.append(cos_sim)

    if not cosine_values:
        return None

    cosine_tensor = torch.tensor(cosine_values, dtype=torch.float32)
    writer.add_scalar(f"{metric_prefix}_mean", cosine_tensor.mean().item(), global_steps)
    writer.add_scalar(f"{metric_prefix}_min", cosine_tensor.min().item(), global_steps)
    writer.add_scalar(f"{metric_prefix}_max", cosine_tensor.max().item(), global_steps)
    writer.add_scalar(f"{metric_prefix}_std", cosine_tensor.std(unbiased=False).item(), global_steps)
    return cosine_tensor.mean().item()


def log_feature_aggregator_agent_cosine_similarity(
    writer,
    clients,
    global_steps,
    metric_prefix="monitoring/aggregation/agent_cosine",
):
    stream_specs = [
        ("feature", "feature_local", "feature_fused"),
        ("action", "action_local", "action_fused"),
    ]

    for stream_name, local_key, fused_key in stream_specs:
        local_by_agent = {}
        fused_by_agent = {}
        for agent_name, client in clients.items():
            summaries = client.debug_feature_aggregator_feature_summaries()
            local_summary = _mean_summary_tensors(
                summary.get(local_key) for summary in summaries.values()
            )
            fused_summary = _mean_summary_tensors(
                summary.get(fused_key) for summary in summaries.values()
            )
            if local_summary is not None:
                local_by_agent[agent_name] = local_summary
            if fused_summary is not None:
                fused_by_agent[agent_name] = fused_summary

        local_mean = _log_pairwise_cosine_stats(
            writer,
            local_by_agent,
            global_steps,
            f"{metric_prefix}/{stream_name}/local",
        )
        fused_mean = _log_pairwise_cosine_stats(
            writer,
            fused_by_agent,
            global_steps,
            f"{metric_prefix}/{stream_name}/fused",
        )
        if local_mean is not None and fused_mean is not None:
            writer.add_scalar(
                f"{metric_prefix}/{stream_name}/fused_minus_local_mean",
                fused_mean - local_mean,
                global_steps,
            )

def make_sample_fn(args, agents_model, cameras, accelerator, deterministic=True):
    def _resize(img, size=128):
        # img = img.unsqueeze(0)          # (1, C, H, W)
        img = F.interpolate(
            img,
            size=size,
            mode='bilinear',
            # align_corners=align_corners if mode != "nearest" else None,
        )
        return img
    device = accelerator.device
    # device 指将obs移动到什么device, 不是obs是什么device
    def sample_fn(obs):
        if isinstance(obs['rgb'], np.ndarray):
            rgb = torch.from_numpy(obs['rgb'] / 255.0).permute(0, 3, 1, 2).float()
            global_state = torch.from_numpy(obs['global_state'])
            agent_states = {}
            for k, state in obs.items():
                if k not in ['rgb', 'global_state']:
                    agent_states[k] = torch.from_numpy(state)
            
        else:
            rgb = obs["rgb"].permute(0, 3, 1, 2).float() / 255.0  # (env, H, W, 3)
            global_state = obs['global_state']
            agent_states = {}
            for k, state in obs.items():
                if k not in ['rgb', 'global_state']:
                    agent_states[k] = state

        rgb = _resize(rgb).to(device)
        for k in agent_states:
            agent_states[k] = agent_states[k].to(device)
        global_state = global_state.to(device)

        batch = {
            'rgb': rgb,
            'global_state': global_state,
        }
        batch.update(agent_states)

        actions = agents_model.get_action(batch, deterministic=deterministic)
        return actions

    return sample_fn

def get_agent_info(args, env_kwargs, agent_obs_rules, collate_fn):
    test_env = make_eval_envs(
        env_id=args.task_name,
        num_envs=1,
        sim_backend='gpu',
        env_kwargs=env_kwargs,
        wrappers=[lambda env: FlattenRGBObservationWrapperForMARL(env, agent_obs_rules=agent_obs_rules)],
    )
    obs, _ = test_env.reset()
    batch = collate_fn(obs)
    agent_names = list(agent_obs_rules.keys())
    state_dim = batch[f'agent_states_{agent_names[0]}'].shape[-1]
    global_state_dim = batch['global_state'].shape[-1]
    action_dim = test_env.unwrapped.agent.agents_dict[agent_names[0]].single_action_space.shape[0]
    # 检查agent是否同质化
    for agent_name in agent_names:
        if batch[f'agent_states_{agent_name}'].shape[-1] != state_dim:
            raise ValueError(f"Agent {agent_name} has different state dimension {batch[f'agent_states_{agent_name}'].shape[-1]} than other agents {state_dim}")
        if test_env.unwrapped.agent.agents_dict[agent_name].single_action_space.shape[0] != action_dim:
            raise ValueError(f"Agent {agent_name} has different action dimension {test_env.unwrapped.agent.agents_dict[agent_name].single_action_space.shape[0]} than other agents {action_dim}")
        
    infos = {
        "state_dim": state_dim,
        "global_state_dim": global_state_dim,
        "action_dim": action_dim,
        "agent_names": agent_names,
    }
    return infos

def make_envs_for_env_kwargs(args, ckpt, env_kwargs_update_id, ori_env_kwargs: dict, agent_obs_rules, device):
    tmp_env_kwargs = ori_env_kwargs.copy()
    env_kwargs_update = env_kwargs_list[env_kwargs_update_id].copy()
    env_name = env_kwargs_update['name']
    del env_kwargs_update['name']
    tmp_env_kwargs.update(**env_kwargs_update)
    env_kwargs_for_eval = tmp_env_kwargs.copy()
    env_kwargs_for_eval.pop('sim_backend')
    tmp_env_kwargs['render_mode'] = 'none'
    env_kwargs_for_eval['shader_dir'] = 'default'
    eval_envs = make_eval_envs(
        env_id=args.task_name,
        num_envs=args.num_eval_envs,
        sim_backend='gpu',
        env_kwargs=env_kwargs_for_eval,
        video_dir=os.path.join(f'{ckpt["video_dir"]}_train', env_name) if not args.evaluate_mode else os.path.join(f'{ckpt["video_dir"]}_eval', env_name),
        wrappers=[
            lambda env: FlattenRGBObservationWrapperForMARL(env, agent_obs_rules=agent_obs_rules),
        ],
    )

    _, _ = eval_envs.reset(seed=args.seed)
    envs = gym.make(args.task_name, num_envs=args.num_envs, **tmp_env_kwargs)
    envs = FlattenRGBObservationWrapperForMARL(envs, agent_obs_rules=agent_obs_rules)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=args.ignore_partial_reset, record_metrics=True)
    
    return envs, eval_envs

# =====================================================
# Main
# =====================================================
def main(args):
    aggregator_lr = args.aggregator_lr if args.aggregator_lr is not None else args.lr * 0.1
    def maybe_switch_envs():
        nonlocal envs, eval_envs, next_obs, next_done, current_env_index, current_env_id
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

        previous_env_id = current_env_id
        current_env_index = scheduled_env_index
        current_env_id = continual_env_schedule.env_kwarg_list[current_env_index]['name']
        print(
            f"switching env from {previous_env_id} to {current_env_id} "
            f"at elapsed={elapsed_minutes:.2f} minutes"
        )
        envs.close()
        eval_envs.close()
        envs, eval_envs = make_envs_for_env_kwargs(
            args,
            ckpt,
            current_env_index,
            env_kwargs,
            agent_obs_rules,
            device
        )
        next_obs, _ = envs.reset(seed=args.seed)
        next_done = torch.zeros(args.num_envs, device=device)
        return True, False, elapsed_minutes
    
    ckpt = resolve_ckpt_dir(args, model_name)
    step_infos = get_step_infos(args)
    env_kwargs = {
      "obs_mode": "rgb+state_dict",
      "control_mode": "pd_joint_delta_pos",
      "render_mode": "rgb_array",
      "reward_mode": "normalized_dense",
      "shader_dir": "minimal",
      "sim_backend": "physx_cuda",
      "max_episode_steps": args.max_episode_steps,
    }
    env_kwargs_for_init = env_kwargs.copy()
    env_kwargs_for_init.pop('sim_backend')
    env_kwargs_for_init['shader_dir'] = 'default'
    current_env_index = 0
    current_env_id = "ObjectScaleUp1p2"
    continual_env_schedule = build_continual_env_schedule(args, env_kwargs_list)

    agent_obs_rules = {
        "panda_wristcam-0": ["cube_pose", "cube_to_goal_pos", "left_arm_tcp_to_cube_pos", "left_arm_tcp", "qpos", "qvel", "stage"],
        "panda_wristcam-1": ["cube_pose", "cube_to_goal_pos", "right_arm_tcp_to_cube_pos", "right_arm_tcp", "qpos", "qvel", "stage"],
    }
    layers_name_of_head = ['critic'] + [f'actor_heads.{name}' for name in agent_obs_rules.keys()] + [f'actor_logstd.{name}' for name in agent_obs_rules.keys()]
    
    tensorboard_port = 6007
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = not args.ignore_torch_deterministic
    accelerator = Accelerator(mixed_precision="bf16" if args.use_amp else "no")
    device = accelerator.device

    if accelerator.is_main_process:
        writer = SummaryWriter(ckpt["log_dir"])
    else:
        writer = None

    assert args.pretrained_agent_path is not None

    infos = get_agent_info(args, env_kwargs_for_init, agent_obs_rules, make_collate_fn(args, cameras, device))
    agent = process_agent(args, infos, env_kwargs_for_init, agent_obs_rules, device)

    if os.path.exists(ckpt["latest_agent"]):
        print(f"[Train] Resume agent from {ckpt['latest_agent']}")
        agent_dict = torch.load(ckpt["latest_agent"], map_location="cpu")
        agent.load_state_dict(agent_dict)
    elif os.path.exists(ckpt["actor_path"]):
        print(f"[Train] load actor from {ckpt['actor_path']}")
        agent.load_actor(ckpt['actor_path'])
    elif os.path.exists(ckpt['ft_agent_path']):
        print(f"[Train] load ft agent from {ckpt['ft_agent_path']}")
        agent_dict = torch.load(ckpt['ft_agent_path'], map_location="cpu")
        agent.load_state_dict(agent_dict)
        # agent.reset_value_head()
    if args.reinit_head:
        print("reinitialize head layers")
        # print(agent)
        for layer_name in layers_name_of_head:
            # print(layer_name)
            if get_module(agent, layer_name) is None:
                continue
            for m in get_module(agent, layer_name).modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()

    if not args.baseline:
        clients = {
                name : ClientForMultiAgent(
                    name=name,
                    large_model=agent,
                    layer_name_of_output_features=f'actor_feature_placeholders.{name}',
                    local_feature_dim=agent.actor_heads[name][0].raw_linear.in_features,
                device=device,
                local_action_dim=agent.actor_heads[name][-1].out_features,
                actor_layer_name=f'actor_heads.{name}',
                max_episode_steps=args.max_episode_steps,
                feature_aggregator_attention_num_heads=args.feature_aggregator_attention_num_heads,
                feature_aggregator_gate_type=args.feature_aggregator_gate_type,
                feature_aggregator_gate_activation=args.feature_aggregator_gate_activation,
                    feature_aggregator_norm_type=args.feature_aggregator_norm_type,
                    feature_aggregator_feature_gate_open_max=args.feature_aggregator_feature_gate_open_max,
                    feature_aggregator_action_gate_open_max=args.feature_aggregator_action_gate_open_max,
                    feature_aggregator_remote_dropout_prob=args.feature_aggregator_remote_dropout_prob,
                    feature_aggregator_remote_noise_std=args.feature_aggregator_remote_noise_std,
                    feature_aggregator_remote_stale_shift_max=args.feature_aggregator_remote_stale_shift_max,
                    feature_selector_topk_trajectories=args.feature_selector_topk_trajectories,
                    feature_selector_temporal_pool_steps=args.feature_selector_temporal_pool_steps,
                    feature_selector_strategy=args.feature_selector_strategy,
                    feature_selector_alpha=0.2,
                )
                for name in agent_obs_rules.keys()
            }
        
        client_infos = {}
        for name in agent_obs_rules.keys():
            clients[name].load_feature_aggregators(os.path.join(args.pretrained_agent_path, f'latest_ag_{name}.pt'))
            # clients[name].load_feature_aggregators(os.path.join(args.pretrained_agent_path, f'best_ag_{name}.pt'))
            client_infos[name] = clients[name].before_training_start(agent)
        for name_i, client_recv in clients.items():
            for name_j, client_send_info in client_infos.items():
                if name_i != name_j:
                    client_recv.add_feature_aggregator(name_j, client_send_info)
    
    training_feature_aggregator_modules = []

    head_lr = args.head_lr if args.head_lr is not None else args.lr
    encoder_lr = args.encoder_lr if args.encoder_lr is not None else (head_lr if head_lr > 0 else args.lr)
    encoder_prefixes = ("rgb_encoder.", "actor_state_encoder.", "critic_state_encoder.")
    head_trainable_parameters = []
    head_trainable_parameter_names = []
    encoder_trainable_parameters = []
    encoder_trainable_parameter_names = []
    for n, p in agent.named_parameters():
        if head_lr > 0 and any([n.startswith(layer_name) for layer_name in layers_name_of_head]):
            p.requires_grad = True
            head_trainable_parameters.append(p)
            head_trainable_parameter_names.append(n)
        elif any(n.startswith(prefix) for prefix in encoder_prefixes):
            p.requires_grad = True
            encoder_trainable_parameters.append(p)
            encoder_trainable_parameter_names.append(n)
        else:
            p.requires_grad = False

    if head_lr > 0:
        print(f"Head parameters trainable (lr={head_lr}): {head_trainable_parameter_names}")
    if len(encoder_trainable_parameters) > 0:
        print(f"Encoder parameters trainable (lr={encoder_lr}): {encoder_trainable_parameter_names}")
    base_trainable_parameters = head_trainable_parameters + encoder_trainable_parameters
    if head_lr <= 0 and len(encoder_trainable_parameters) == 0:
        print(f"All model parameters frozen. Only feature aggregators will be trained.")
    else:
        print(f"Base trainable parameter count: {len(base_trainable_parameters)}")

    # Optimizer: head params with small LR, aggregator params added later with full LR
    optimizer_param_groups = []
    if len(head_trainable_parameters) > 0:
        optimizer_param_groups.append({'params': head_trainable_parameters, 'lr': head_lr})
    if len(encoder_trainable_parameters) > 0:
        optimizer_param_groups.append({'params': encoder_trainable_parameters, 'lr': encoder_lr})

    if len(optimizer_param_groups) > 0:
        optimizer = optim.Adam(optimizer_param_groups, eps=1e-5)
    else:
        optimizer = optim.Adam(
            [{'params': [torch.zeros(1, requires_grad=True)], 'lr': args.lr}], eps=1e-5  # placeholder
        )

    sta_steps = global_steps = 0
    start_time = time.time()
    if os.path.exists(ckpt["latest_opt"]):
        print(f"[Train] Resume optimizer from {ckpt['latest_opt']}")
        resume_opt = torch.load(ckpt["latest_opt"], map_location="cpu")
        optimizer.load_state_dict(resume_opt['opt'])
        sta_steps = global_steps = resume_opt['step']

    agent, optimizer = accelerator.prepare(agent, optimizer)

    if args.reset_logstd:
        print(f"[Train] Reset actor log std to -0.5")
        agent.reset_logstd(-0.5)

    def set_requires_grad(parameters, requires_grad: bool):
        for p in parameters:
            p.requires_grad = requires_grad

    best_score = -1
    metrics_log = []

    if args.evaluate_mode:
        # if args.eval_agent_dir is None:
        #     raise ValueError("Please provide --eval-agent-dir for evaluation mode")
        
        # eval_envs = make_eval_envs(
        #     env_id=args.task_name,
        #     num_envs=args.num_eval_envs,
        #     sim_backend='gpu',
        #     # action_repeat=control_freq // real_control_freq,
        #     env_kwargs=env_kwargs_for_eval,
        #     video_dir=f'{ckpt["video_dir"]}_train' if not args.evaluate_mode else os.path.join(args.eval_agent_dir, 'videos_eval'),
        #     wrappers=[
        #         lambda env: FlattenRGBObservationWrapperForMARL(env, agent_obs_rules=agent_obs_rules),
        #         # lambda env: ActionRepeatWrapperForPosControl(env, real_freq=real_control_freq, delta_pos=False)
        #     ],
        # )
        
        # agent.load_state_dict(torch.load(os.path.join(args.eval_agent_dir, "best_agent.pt"), map_location="cpu"))
        # # agent.load_state_dict(torch.load(os.path.join(args.eval_agent_dir, "latest_agent.pt"), map_location="cpu"))
        # agent = accelerator.prepare(agent)
        # print("[Evaluate] Start evaluation only mode")
        # agent.eval()
        # eval_metrics = evaluate(
        #     n=100,
        #     # n=1,
        #     sample_fn=make_sample_fn(args, agent, cameras, accelerator, deterministic=True),
        #     eval_envs=eval_envs,
        # )
        # for k, v in eval_metrics.items():
        #     mean = v.mean()
        #     print(f"eval_{k}_mean={mean}")
        # dump_json(os.path.join(ckpt['root_dir'], 'eval_metrics.json'), {k: float(v.mean()) for k, v in eval_metrics.items()})
        return

    pbar = tqdm(total=step_infos['total_steps'], initial=global_steps, ascii=True)
    writer = SummaryWriter(ckpt["log_dir"], purge_step=global_steps)
    print(f"[TensorBoard] Logging to {ckpt['log_dir']}")
    print(f"[TensorBoard] Using `tensorboard --logdir {ckpt['log_dir']} --port {tensorboard_port}` to show the logs")

    sparsity_list = np.linspace(0, args.max_sparsity, 3).tolist()[0: 3]

    envs, eval_envs = make_envs_for_env_kwargs(
        args,
        ckpt,
        0,
        env_kwargs,
        agent_obs_rules,
        device
    )
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    if os.path.exists(ckpt["metrics"]):
        metrics_log = load_json(ckpt["metrics"])
    resume_skip = True if args.resume_dir is not None else False
    # resume_skip = True

    if args.minibatch_size == 0:
        args.minibatch_size = step_infos['rollot_steps'] // args.num_minibatch // args.grad_accum_steps

    training_start_time = time.monotonic()

    while global_steps < step_infos['total_steps']:
        switched_env, should_stop_for_schedule, elapsed_minutes = maybe_switch_envs()
        # if switched_env and not args.baseline:
        #     for client in clients.values():
        #         client.reset_feature_selector_cache()
        #         client.clear_messages()
        if writer is not None and elapsed_minutes is not None:
            writer.add_scalar("time/elapsed_minutes", elapsed_minutes, global_steps)
            writer.add_scalar("continual/current_env_index", current_env_index, global_steps)
        if should_stop_for_schedule:
            print(
                f"Reached continual schedule end at elapsed={elapsed_minutes:.2f} minutes, stopping."
            )
            break

        if args.max_time is not None:
            elapsed_minutes = (time.monotonic() - training_start_time) / 60.0
            if elapsed_minutes >= args.max_time:
                print(f"Reached max_time={args.max_time} minutes, stopping.")
                break

        agent.eval()
        if not args.baseline:
            for client in clients.values():
                client.eval()

        if not resume_skip and global_steps % (step_infos['save_interval_steps']) == 0:
            avg_score = 0.

            torch.save(agent.state_dict(), ckpt['latest_agent'])
            torch.save({'opt': optimizer.state_dict(), 'step': global_steps}, ckpt["latest_opt"])
            if not args.baseline:
                for name in agent_obs_rules.keys():
                    clients[name].save_feature_aggregators(os.path.join(ckpt['root_dir'], f'best_ag_{name}.pt'))

            test_sparsity = 0.
            set_sparsity(agent, test_sparsity)
            print(f'Sparsity {test_sparsity:.2f} :')
            # ---------------- evaluate ----------------
            eval_metrics = evaluate(
                n=1,
                sample_fn=make_sample_fn(args, agent, cameras, accelerator),
                eval_envs=eval_envs,
            )
            for k, v in eval_metrics.items():
                mean = v.mean()
                # writer.add_scalars(f"eval/{k}", {test_sparsity : mean}, global_steps)
                writer.add_scalar(f"eval/{k}", mean, global_steps)
                print(f"eval_{k}_mean={mean}")
            score = eval_metrics.get(
                "success_rate", eval_metrics[list(eval_metrics.keys())[0]]
            ).mean()
            
            avg_score += score

            avg_score /= 1
            pbar.set_postfix(eval_score=avg_score)
            metrics_log.append(dict(step=global_steps, score=float(avg_score)))
            dump_json(ckpt["metrics"], metrics_log)
            if avg_score >= best_score:
                best_score = avg_score
                torch.save(agent.state_dict(), ckpt['best_agent'])
                if not args.baseline:
                    for name in agent_obs_rules.keys():
                        clients[name].save_feature_aggregators(os.path.join(ckpt['root_dir'], f'best_ag_{name}.pt'))
                print(f"[Eval] New best model saved (score={avg_score:.3f})")

        resume_skip = False
        # cur_sparsity = random.uniform(sparsity_list[0], sparsity_list[-1])
        cur_sparsity = 0
        set_sparsity(agent, cur_sparsity)

        if args.dynamic_lr:
            pass
            # adjust_lr_cnn(optimizer, global_steps, step_infos, init_lr)
        if args.dynamic_clip:
            frac = 1.0 - (global_steps / step_infos["total_steps"])
            args.clip_eps = 0.1 + (args.clip_eps - 0.1) * frac
        if args.dynamic_ent_coef:
            frac = 1.0 - (global_steps / step_infos["total_steps"])
            args.ent_coef = args.ent_coef / 10 + (args.ent_coef * 0.9) * frac

        if not args.baseline:
            for client in clients.values():
                client.train()

        collate_fn = make_collate_fn(args, cameras, device)
        rollout = collect_rollout(
            args=args, 
            agent=agent, 
            collate_fn=collate_fn, 
            accelerator=accelerator,
            envs=envs,
            next_obs=next_obs,
            next_done=next_done,
            writer=writer,
            global_step=global_steps,
            clients=clients if not args.baseline else None
        )

        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, final_val_buf, next_obs, next_done = rollout

        adv_buf, ret_buf = compute_gae(rew_buf, done_buf, val_buf, final_val_buf, next_obs, next_done, agent, collate_fn, args, accelerator)

        data = (
            obs_buf.reshape((-1,)),
            act_buf.reshape((-1,)),
            logp_buf.reshape((-1,)),
            adv_buf.reshape(-1),
            ret_buf.reshape(-1),
            val_buf.reshape(-1),
        )
        agent.train()
        
        if not args.baseline:
            all_aggregator_params = []
            for agent_name, client in clients.items():
                for _, fa in client.feature_aggregators.items():
                    for p in fa.module.parameters():
                        all_aggregator_params.append(p)

            has_active_aggregator = any(
                fa.remote_features is not None
                for client in clients.values()
                for fa in client.feature_aggregators.values()
            )
            active_head_trainable = head_lr > 0 and len(head_trainable_parameters) > 0
            active_encoder_trainable = encoder_lr > 0 and len(encoder_trainable_parameters) > 0
            has_active_base_training = active_head_trainable or active_encoder_trainable
            update_mode = "joint" if args.joint_policy_aggregator_update else "separate"

            print(
                f'Clients updating (mode={update_mode}, heads_lr={head_lr}, '
                f'encoder_lr={encoder_lr}, agg_lr={aggregator_lr}, '
                f'agg_active={has_active_aggregator})...'
            )

        if args.baseline:
            stats = mappo_update_on_policy(
                args,
                agent,
                optimizer,
                data,
                collate_fn,
                accelerator,
                f"MAPPO Training (Sparsity:{cur_sparsity:.2f})",
                writer,
                -1,
            )
        elif args.joint_policy_aggregator_update:
            set_requires_grad(head_trainable_parameters, active_head_trainable)
            set_requires_grad(encoder_trainable_parameters, active_encoder_trainable)
            set_requires_grad(all_aggregator_params, not args.not_train_aggregator)

            if has_active_aggregator and len(all_aggregator_params) > 0 and not args.not_train_aggregator:
                stats = mappo_update_on_policy_ag(
                    args,
                    agent,
                    optimizer,
                    data,
                    collate_fn,
                    accelerator,
                    f"MAPPO Joint Training (Sparsity:{cur_sparsity:.2f})",
                    clients,
                    writer,
                    -1,
                    target_kl_override=args.joint_target_kl,
                )
            elif has_active_base_training:
                print('Clients no active aggregator, fallback to base MAPPO update')
                stats = mappo_update_on_policy(
                    args,
                    agent,
                    optimizer,
                    data,
                    collate_fn,
                    accelerator,
                    f"MAPPO Training (Sparsity:{cur_sparsity:.2f})",
                    writer,
                    -1,
                )
            else:
                print('Clients no active aggregator and base model frozen, skipping update')
                stats = dict(
                    policy_loss=torch.tensor(0.0),
                    value_loss=torch.tensor(0.0),
                    entropy=torch.tensor(0.0),
                    approx_kl=torch.tensor(0.0),
                    old_approx_kl=torch.tensor(0.0),
                    clip_frac=torch.tensor(0.0),
                )
        elif has_active_base_training:
            set_requires_grad(all_aggregator_params, False)
            stats = mappo_update_on_policy(
                args,
                agent,
                optimizer,
                data,
                collate_fn,
                accelerator,
                f"MAPPO Training (Sparsity:{cur_sparsity:.2f})",
                writer,
                -1,
            )
        else:
            stats = dict(
                policy_loss=torch.tensor(0.0),
                value_loss=torch.tensor(0.0),
                entropy=torch.tensor(0.0),
                approx_kl=torch.tensor(0.0),
                old_approx_kl=torch.tensor(0.0),
                clip_frac=torch.tensor(0.0),
            )
        
        if not args.baseline:
            if not args.joint_policy_aggregator_update:
                if has_active_aggregator and len(all_aggregator_params) > 0 and not args.not_train_aggregator:
                    set_requires_grad(head_trainable_parameters, False)
                    set_requires_grad(encoder_trainable_parameters, False)
                    set_requires_grad(all_aggregator_params, True)
                    stats = mappo_update_on_policy_ag(
                        args,
                        agent,
                        optimizer,
                        data,
                        collate_fn,
                        accelerator,
                        f"MAPPO Aggregator Training (Sparsity:{cur_sparsity:.2f})",
                        clients,
                        writer,
                        -1,
                    )
                    set_requires_grad(head_trainable_parameters, head_lr > 0)
                    set_requires_grad(encoder_trainable_parameters, active_encoder_trainable)
                else:
                    if not has_active_aggregator:
                        print(f'Clients no active aggregator, skipping aggregator update')
                    # Set defaults for logging
    
            ag_data_infos = {}
            for client_name, client in clients.items():
                ag_data_infos[client_name] = client.export_feature_and_action()
            log_uploaded_q_task_stats(writer, ag_data_infos, global_steps)
            log_agent_feature_cosine_similarity(writer, ag_data_infos, global_steps)

            for client_name, client_recv in clients.items():
                for client_send_name, ag_data in ag_data_infos.items():
                    if client_name != client_send_name:
                        client_recv.receive_feature_and_action(client_send_name, ag_data)
            
            if not args.not_train_aggregator:
                for client_name, client in clients.items():
                    feature_aggregators_parameters = client.get_feature_aggregators_parameters()
                    for client_id, fap in feature_aggregators_parameters.items():
                        n_name = f'{client_name}_from_{client_id}'
                        if n_name not in training_feature_aggregator_modules:
                            training_feature_aggregator_modules.append(n_name)
                            fap_list = list(fap)  # must convert to list BEFORE iterating, generator can only be consumed once
                            for p in fap_list:
                                p.requires_grad = True
                            optimizer.add_param_group({'params': fap_list, 'lr': aggregator_lr, 'eps': 1e-5})

                    feature_aggregators_gate_g = client.debug_feature_aggregators()
                    gate_g_strs = []
                    for client_id, gate_info in feature_aggregators_gate_g.items():
                        for stream_name, gate_g in gate_info.items():
                            if gate_g is None:
                                continue
                            metric_prefix = (
                                f"monitoring/aggregation/gate_g/{client_name}/from_{client_id}/{stream_name}"
                            )
                            writer.add_histogram(metric_prefix, gate_g, global_steps)
                            writer.add_scalar(f"{metric_prefix}_mean", gate_g.mean(), global_steps)
                            writer.add_scalar(f"{metric_prefix}_std", gate_g.std(), global_steps)
                            gate_g_strs.append(f"{client_id}.{stream_name}={gate_g.mean().item():.4f}")
                    if gate_g_strs:
                        print(f'Client {agent_name} gate_g_mean: {", ".join(gate_g_strs)}')

            log_feature_aggregator_agent_cosine_similarity(writer, clients, global_steps)

        global_steps += step_infos['rollot_steps']
        pbar.update(step_infos['rollot_steps'])

        y_pred, y_true = val_buf.flatten(0, 1).cpu().numpy(), ret_buf.flatten(0, 1).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = (global_steps - sta_steps) / (time.time() - start_time)
        writer.add_scalar("charts/SPS", sps, global_steps)
        writer.add_scalar("loss/policy", stats["policy_loss"], global_steps)
        writer.add_scalar("loss/value", stats["value_loss"], global_steps)
        writer.add_scalar("loss/entropy", stats["entropy"], global_steps)
        writer.add_scalar("loss/approx_kl", stats["approx_kl"], global_steps)
        writer.add_scalar("loss/old_approx_kl", stats["old_approx_kl"], global_steps)
        writer.add_scalar("loss/clip_frac", stats["clip_frac"], global_steps)
        writer.add_scalar("loss/explained_var", explained_var, global_steps)
    writer.close()
    torch.save(agent.state_dict(), ckpt['latest_agent'])
    torch.save({'opt': optimizer.state_dict(), 'step': global_steps}, ckpt["latest_opt"])
# batchsize=num_envs * rollout_steps
# minibatch_size = batchsize // num_minibatch
# 若envs中存在好样本，minibatch_size应该足够大以覆盖好样本，但过大会稀释好样本的影响
# 若资源不够，可以采用grad_accum_steps来累积梯度，相当于增大了minibatch_size
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--actor-ckpt-path", type=str, default='ckpt/PickCube-v1/ours/toy_cnn/pretrain_large_model/20260121-092802/checkpoints')
    parser.add_argument("--actor-ckpt-path", type=str, default=None)
    parser.add_argument("--ft-agent-path", type=str, default=None)
    parser.add_argument("--task-name", type=str, default="TwoRobotPickCube-v2")
    parser.add_argument("--seed", type=int, default=1788)
    parser.add_argument("--total-steps", type=int, default=50_000_000)
    parser.add_argument("--critic-warmup-rollouts", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-eval-envs", type=int, default=50)
    parser.add_argument("--ignore-partial-reset", action="store_true")
    parser.add_argument("--ignore-torch-deterministic", action="store_true")
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--num_minibatch", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--reward-scale", type=int, default=1)
    parser.add_argument("--rollout-minibatch-size", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--dynamic-lr", action="store_true")
    parser.add_argument("--dynamic-clip", action="store_true")
    parser.add_argument("--dynamic-ent-coef", action="store_true")
    parser.add_argument("--normalize-state", action="store_true")
    parser.add_argument("--not-normalize-adv", action="store_true")
    parser.add_argument("--reset-logstd", action="store_true")
    parser.add_argument("--finite-horizon-gae", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--aggregator-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--encoder-lr", type=float, default=None)
    # g 0.956 l 0.966
    parser.add_argument("--gamma", type=float, default=0.956)
    parser.add_argument("--gae-lambda", type=float, default=0.966)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--gate-reg-coef", type=float, default=0.0)
    parser.add_argument("--gate-target-mean", type=float, default=0.2)
    parser.add_argument("--gate-std-coef", type=float, default=0.0)
    parser.add_argument("--feature-gate-reg-coef", type=float, default=None)
    parser.add_argument("--feature-gate-target-mean", type=float, default=0.2)
    parser.add_argument("--feature-gate-std-coef", type=float, default=0.0)
    parser.add_argument("--feature-gate-quality-coef", type=float, default=1.0)
    parser.add_argument("--action-gate-reg-coef", type=float, default=None)
    parser.add_argument("--action-gate-target-mean", type=float, default=0.1)
    parser.add_argument("--action-gate-std-coef", type=float, default=0.0)
    parser.add_argument("--action-gate-quality-coef", type=float, default=1.5)
    parser.add_argument("--feature-consistency-coef", type=float, default=0.5)
    parser.add_argument("--action-consistency-coef", type=float, default=1.0)
    parser.add_argument("--clip-vloss", action="store_true")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.2)
    parser.add_argument("--aggregator-target-kl", type=float, default=0.2)
    parser.add_argument(
        "--joint-target-kl",
        type=float,
        default=0.5,
        help="KL threshold used only in joint policy+aggregator update mode"
    )
    parser.set_defaults(joint_policy_aggregator_update=True)
    parser.add_argument(
        "--joint-policy-aggregator-update",
        dest="joint_policy_aggregator_update",
        action="store_true",
        help="run a single PPO pass that jointly updates policy encoder/head and feature aggregator"
    )
    parser.add_argument(
        "--separate-policy-aggregator-update",
        dest="joint_policy_aggregator_update",
        action="store_false",
        help="keep the old two-stage update path: base PPO first, aggregator PPO second"
    )
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--save-dir", type=str, default="ckpt")
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--pretrained-agent-path", type=str, default="ckpt/TwoRobotPickCube-v2_ag/mappo/pandas_pandas/toy_cnn/20260607-043942")
    parser.add_argument("--max-sparsity", type=float, default=0.8)
    
    parser.add_argument("--save-interval-per-rollout", type=int, default=2)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--evaluate-mode", action="store_true")
    parser.add_argument("--eval-agent-dir", type=str, default=None)
    
    parser.add_argument("--robot-name", type=str, default="pandas_pandas")
    parser.add_argument("--reinit-head", action="store_true")
    parser.add_argument("--baseline", action="store_true")

    parser.add_argument(
        "--feature-selector-topk-trajectories",
        type=int,
        default=4,
        help="number of highest-return trajectories to upload as remote features"
    )

    parser.add_argument(
        "--feature-selector-temporal-pool-steps",
        type=int,
        default=16,
        help="temporal pooling target length for uploaded trajectories; set None to disable"
    )

    parser.add_argument(
        "--feature-selector-strategy",
        type=str,
        default="topk_return",
        choices=["topk_return", "random", "return_span"],
        help="trajectory selection strategy"
    )

    parser.add_argument(
        "--feature-aggregator-attention-num-heads",
        type=int,
        default=1,
        help="number of attention heads used in feature aggregator"
    )

    parser.add_argument(
        "--feature-aggregator-gate-type",
        type=str,
        default="single-layer",
        choices=["single-layer", "two-layers"],
        help="gate architecture in feature aggregator"
    )

    parser.add_argument(
        "--feature-aggregator-gate-activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "silu", "tanh"],
        help="activation used inside two-layers gate"
    )

    parser.add_argument(
        "--feature-aggregator-norm-type",
        type=str,
        default="layernorm",
        choices=["none", "layernorm"],
        help="normalization used inside feature aggregator"
    )

    parser.add_argument(
        "--feature-aggregator-feature-gate-open-max",
        type=float,
        default=0.25,
        help="upper bound used by q-based supervision for feature-stream gate openness"
    )

    parser.add_argument(
        "--feature-aggregator-action-gate-open-max",
        type=float,
        default=0.10,
        help="upper bound used by q-based supervision for action-stream gate openness"
    )

    parser.add_argument(
        "--feature-aggregator-remote-dropout-prob",
        type=float,
        default=0.05,
        help="lightweight message-level dropout for remote feature/action messages during aggregator training"
    )

    parser.add_argument(
        "--feature-aggregator-remote-noise-std",
        type=float,
        default=0.0,
        help="optional Gaussian noise added to remote features/actions during aggregator training"
    )

    parser.add_argument(
        "--feature-aggregator-remote-stale-shift-max",
        type=int,
        default=0,
        help="optional maximum temporal stale shift applied to remote trajectories during aggregator training"
    )

    parser.add_argument(
        "--not-train-aggregator",
        action="store_true"
    )

    parser.add_argument(
        "--env-change-time-points",
        type=str,
        default="[21,42,63,84,105,126,147,168,189,210]",
        help="change env time points(min)"
    )

    parser.add_argument(
        "--max-time",
        type=int,
        default=None
    )
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # 启动监控线程（守护线程，不阻塞主程序）
    main(parse_args())
