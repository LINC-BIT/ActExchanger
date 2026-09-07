from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image


_DEBUG_RGB_SAVED = False


def _as_tensor(value: Any, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
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


def maybe_save_debug_rgb(obs: Dict[str, Any], output_path: str = "debug_place_cucumber_rgb.png") -> None:
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
    qvel = _ensure_2d(_as_tensor(obs["agent"][agent_name]["qvel"], dtype=torch.float32))

    if agent_name.endswith("-0"):
        return torch.cat(
            [
                qpos,
                qvel,
                _extra(obs, "center_tcp", batch_size),
                _extra(obs, "center_tcp_q", batch_size),
                _extra(obs, "lid_pos", batch_size),
                _extra(obs, "lid_handle_pos", batch_size),
                _extra(obs, "lid_qpos", batch_size),
                _extra(obs, "lid_qvel", batch_size),
                _extra(obs, "center_tcp_to_lid_handle", batch_size),
            ],
            dim=-1,
        )

    is_left_cucumber = agent_name.endswith("-1")
    prefix = "left" if is_left_cucumber else "right"
    cucumber_idx = "1" if is_left_cucumber else "2"
    return torch.cat(
        [
            qpos,
            qvel,
            _extra(obs, f"{prefix}_tcp", batch_size),
            _extra(obs, f"{prefix}_tcp_q", batch_size),
            _extra(obs, f"{prefix}_grip_line_axis", batch_size),
            _extra(obs, f"cucumber{cucumber_idx}_pos", batch_size),
            _extra(obs, f"cucumber{cucumber_idx}_q", batch_size),
            _extra(obs, f"cucumber{cucumber_idx}_long_axis", batch_size),
            _extra(obs, "pot_pos", batch_size),
            _extra(obs, f"{prefix}_tcp_to_cucumber{cucumber_idx}", batch_size),
            _extra(obs, f"cucumber{cucumber_idx}_to_pot", batch_size),
        ],
        dim=-1,
    )


def extract_global_state_from_obs(obs: Dict[str, Any], agent_names: List[str]) -> torch.Tensor:
    parts = []
    batch_size = None
    for name in agent_names:
        qpos = _ensure_2d(_as_tensor(obs["agent"][name]["qpos"], dtype=torch.float32))
        batch_size = qpos.shape[0]
        parts.extend([qpos, _ensure_2d(_as_tensor(obs["agent"][name]["qvel"], dtype=torch.float32))])
    for key in [
        "center_tcp",
        "left_tcp",
        "right_tcp",
        "center_tcp_q",
        "left_tcp_q",
        "right_tcp_q",
        "pot_pos",
        "lid_pos",
        "lid_handle_pos",
        "lid_qpos",
        "lid_qvel",
        "cucumber1_pos",
        "cucumber2_pos",
        "cucumber1_q",
        "cucumber2_q",
        "cucumber1_long_axis",
        "cucumber2_long_axis",
        "left_grip_line_axis",
        "right_grip_line_axis",
        "left_tcp_to_cucumber1",
        "right_tcp_to_cucumber2",
        "center_tcp_to_lid_handle",
        "cucumber1_to_pot",
        "cucumber2_to_pot",
    ]:
        parts.append(_extra(obs, key, batch_size))
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
