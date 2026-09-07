from __future__ import annotations

import math
from typing import Sequence

import numpy as np


MIXED_FIVE_VLA_BACKBONES = (
    "mixed_five_vla_100m",
    "mixed_tiny_vla_smolvla_efficientvla",
)

BASE_LANE_X = -0.45
BASE_CUBE_Y = -0.82
HANDOFF_Y_OFFSETS = (0.38, 0.78, 1.18)
BASE_SO100_YAW = math.pi / 2.0
BASE_XARM6_XY = (-0.98, 0.29)
BASE_HAND_POSE = (-0.37, 0.74, 0.03)


def is_mixed_five_vla_backbone(model_backbone: str) -> bool:
    return str(model_backbone) in MIXED_FIVE_VLA_BACKBONES


def make_pass_object_env_variant(
    name: str,
    *,
    lane_x: float = BASE_LANE_X,
    cube_y: float = BASE_CUBE_Y,
    cube_half_size: float | None = None,
    table_yaw: float = 0.0,
    so100_yaw: float = BASE_SO100_YAW,
    xarm6_xy: tuple[float, float] = BASE_XARM6_XY,
    hand_pose: tuple[float, float, float] = BASE_HAND_POSE,
) -> dict:
    handoff_y_01 = cube_y + HANDOFF_Y_OFFSETS[0]
    handoff_y_12 = cube_y + HANDOFF_Y_OFFSETS[1]
    handoff_y_23 = cube_y + HANDOFF_Y_OFFSETS[2]
    env_kwargs = {
        "name": name,
        "cube_xy": (float(lane_x), float(cube_y)),
        "handoff_xy_01": (float(lane_x), float(handoff_y_01)),
        "handoff_xy_12": (float(lane_x), float(handoff_y_12)),
        "handoff_xy_23": (float(lane_x), float(handoff_y_23)),
        "table_yaw": float(table_yaw),
        "so100_yaw": float(so100_yaw),
        "xarm6_xy": (float(xarm6_xy[0]), float(xarm6_xy[1])),
        "hand_pose": (
            float(hand_pose[0]),
            float(hand_pose[1]),
            float(hand_pose[2]),
        ),
    }
    if cube_half_size is not None:
        env_kwargs["cube_half_size"] = float(cube_half_size)
    return env_kwargs


PASS_OBJECT_ONLINE_HIGH_LOW_PATTERN = [
    make_pass_object_env_variant(
        "XarmBackSoft",
        xarm6_xy=(-1.00, 0.30),
        hand_pose=(-0.365, 0.745, 0.03),
    ),
    make_pass_object_env_variant("CubeScale_1.35x", cube_half_size=0.022 * 1.35),
    make_pass_object_env_variant("YawRightSoft", table_yaw=0.06),
    make_pass_object_env_variant("CubeScale_1.37x", cube_half_size=0.022 * 1.37),
    make_pass_object_env_variant("SO100YawOutSoft", so100_yaw=BASE_SO100_YAW + 0.08),
]

DEFAULT_ONLINE_ENV_KWARGS_LIST = [
    dict(variant)
    for _ in range(2)
    for variant in PASS_OBJECT_ONLINE_HIGH_LOW_PATTERN
]

ONLINE_FAMILY_TRAIN_ENV_KWARGS_LIST = [
    {"name": "Clean"},
    *[dict(variant) for variant in PASS_OBJECT_ONLINE_HIGH_LOW_PATTERN],
]


def sample_pass_object_randomization(
    rng: np.random.Generator,
    *,
    level: str,
) -> tuple[dict, str]:
    level = str(level).strip().lower()
    if level == "clear":
        return {}, "mixed_train_clear"

    if level == "mild":
        lane_x = float(rng.uniform(-0.485, -0.415))
        cube_y = float(rng.uniform(-0.845, -0.795))
        table_yaw = float(rng.uniform(-0.10, 0.10))
        so100_yaw = float(BASE_SO100_YAW + rng.uniform(-0.12, 0.12))
        xarm6_xy = (float(rng.uniform(-1.01, -0.95)), float(rng.uniform(0.27, 0.31)))
        hand_pose = (
            float(rng.uniform(-0.385, -0.355)),
            float(rng.uniform(0.725, 0.755)),
            float(rng.uniform(0.025, 0.040)),
        )
    else:
        lane_x = float(rng.uniform(-0.50, -0.40))
        cube_y = float(rng.uniform(-0.86, -0.78))
        table_yaw = float(rng.uniform(-0.18, 0.18))
        so100_yaw = float(BASE_SO100_YAW + rng.uniform(-0.22, 0.22))
        xarm6_xy = (float(rng.uniform(-1.04, -0.92)), float(rng.uniform(0.25, 0.33)))
        hand_pose = (
            float(rng.uniform(-0.395, -0.345)),
            float(rng.uniform(0.715, 0.765)),
            float(rng.uniform(0.020, 0.045)),
        )

    env_kwargs = make_pass_object_env_variant(
        f"mixed_train_{level}",
        lane_x=lane_x,
        cube_y=cube_y,
        table_yaw=table_yaw,
        so100_yaw=so100_yaw,
        xarm6_xy=xarm6_xy,
        hand_pose=hand_pose,
    )
    env_kwargs.pop("name", None)
    return env_kwargs, f"mixed_train_{level}"


def build_pass_object_toy_cnn_agent_obs_rules(agent_names: Sequence[str]) -> dict[str, list[str]]:
    shared_extra = [
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
    tcp_to_cube_keys = {}
    for name in agent_names:
        if name.startswith("panda"):
            tcp_to_cube_keys[name] = "panda_tcp_to_cube_pos"
        elif name.startswith("so100"):
            tcp_to_cube_keys[name] = "so100_tcp_to_cube_pos"
        elif name.startswith("widowx"):
            tcp_to_cube_keys[name] = "widowx_tcp_to_cube_pos"
        elif name.startswith("xarm6"):
            tcp_to_cube_keys[name] = "xarm6_tcp_to_cube_pos"
        elif "inspire" in name or name.startswith("fixed_inspire_hand"):
            tcp_to_cube_keys[name] = "hand_palm_to_cube_pos"
        else:
            raise KeyError(f"Unsupported agent name for tcp_to_cube mapping: {name}")
    return {
        name: ["qpos", "qvel", tcp_to_cube_keys[name], *shared_extra]
        for name in agent_names
    }
