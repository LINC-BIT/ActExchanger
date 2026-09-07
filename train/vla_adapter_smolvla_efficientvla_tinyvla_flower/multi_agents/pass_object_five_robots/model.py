from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image


_DEBUG_RGB_SAVED = False

SHARED_EXTRA_KEYS = [
    "cube_pos",
    "cube_q",
    "panda_tcp",
    "so100_tcp",
    "widowx_tcp",
    "xarm6_tcp",
    "hand_palm",
    "handoff_target_01",
    "handoff_target_12",
    "handoff_target_23",
    "palm_target",
    "current_stage",
]

AGENT_TCP_TO_CUBE_KEYS = {
    "panda": "panda_tcp_to_cube_pos",
    "so100": "so100_tcp_to_cube_pos",
    "widowx": "widowx_tcp_to_cube_pos",
    "xarm6": "xarm6_tcp_to_cube_pos",
    "fixed_inspire_hand": "hand_palm_to_cube_pos",
}

def _as_tensor(value: Any, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _ensure_2d(tensor: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
    if tensor.ndim == 0:
        if batch_size is None:
            return tensor.reshape(1, 1)
        return tensor.expand(batch_size).reshape(batch_size, 1)
    if tensor.ndim == 1:
        if batch_size is not None and tensor.shape[0] == batch_size:
            return tensor.unsqueeze(-1)
        return tensor.unsqueeze(0)
    return tensor


def _agent_prefix(agent_name: str) -> str:
    for prefix in AGENT_TCP_TO_CUBE_KEYS.keys():
        if agent_name.startswith(prefix):
            return prefix
    raise KeyError(f"Unsupported pass-object agent name: {agent_name}")


def maybe_save_debug_rgb(obs: Dict[str, Any], output_path: str = "debug_pass_object_five_robots_rgb.png") -> None:
    global _DEBUG_RGB_SAVED
    if _DEBUG_RGB_SAVED:
        return
    sensor_data = obs.get("sensor_data", {})
    if "base_camera" not in sensor_data or "rgb" not in sensor_data["base_camera"]:
        return
    rgb = sensor_data["base_camera"]["rgb"]
    if isinstance(rgb, torch.Tensor):
        image = rgb[0] if rgb.ndim == 4 else rgb
        image = image[..., :3].detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    else:
        image = np.asarray(rgb)
        if image.ndim == 4:
            image = image[0]
        image = image[..., :3].astype(np.uint8, copy=False)
    Image.fromarray(image).save(output_path)
    _DEBUG_RGB_SAVED = True


def extract_rgb_batch_from_obs(obs: Dict[str, Any]) -> Union[np.ndarray, torch.Tensor]:
    rgb = obs["sensor_data"]["base_camera"]["rgb"]
    if isinstance(rgb, torch.Tensor):
        return rgb[..., :3]
    return np.asarray(rgb)[..., :3]


def _extra(obs: Dict[str, Any], key: str, batch_size: Optional[int] = None) -> torch.Tensor:
    return _ensure_2d(_as_tensor(obs["extra"][key], dtype=torch.float32), batch_size=batch_size)


def extract_agent_state_from_obs(obs: Dict[str, Any], agent_name: str) -> torch.Tensor:
    qpos = _ensure_2d(_as_tensor(obs["agent"][agent_name]["qpos"], dtype=torch.float32))
    batch_size = qpos.shape[0]
    qvel = _ensure_2d(_as_tensor(obs["agent"][agent_name]["qvel"], dtype=torch.float32), batch_size=batch_size)
    prefix = _agent_prefix(agent_name)
    # Match the legacy pass_object actor-state semantics exactly: the policy
    # consumed tcp_to_cube, not the raw tcp position.
    parts = [qpos, qvel, _extra(obs, AGENT_TCP_TO_CUBE_KEYS[prefix], batch_size)]
    parts.extend(_extra(obs, key, batch_size) for key in SHARED_EXTRA_KEYS)
    return torch.cat(parts, dim=-1)


def extract_global_state_from_obs(obs: Dict[str, Any], agent_names: List[str]) -> torch.Tensor:
    parts = []
    batch_size = None
    for name in agent_names:
        qpos = _ensure_2d(_as_tensor(obs["agent"][name]["qpos"], dtype=torch.float32))
        batch_size = qpos.shape[0]
        qvel = _ensure_2d(_as_tensor(obs["agent"][name]["qvel"], dtype=torch.float32), batch_size=batch_size)
        parts.extend([qpos, qvel])
    parts.extend(_extra(obs, key, batch_size) for key in SHARED_EXTRA_KEYS)
    parts.extend(_extra(obs, AGENT_TCP_TO_CUBE_KEYS[_agent_prefix(name)], batch_size) for name in agent_names)
    return torch.cat(parts, dim=-1)


def build_batch_from_obs(obs: Dict[str, Any], agent_names: List[str]) -> Dict[str, Any]:
    maybe_save_debug_rgb(obs)
    batch = {
        "rgb": extract_rgb_batch_from_obs(obs),
        "global_state": extract_global_state_from_obs(obs, agent_names),
    }
    for name in agent_names:
        batch[f"agent_states_{name}"] = extract_agent_state_from_obs(obs, name)
    return batch
