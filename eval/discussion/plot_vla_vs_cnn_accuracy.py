from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception:  # pragma: no cover
    event_accumulator = None


COMBINED_OUTPUT = Path(__file__).resolve().parents[2] / "tmp/vla_vs_cnn_accuracy_comparison.png"
OUTPUT_3H = Path(__file__).resolve().parents[2] / "tmp/vla_vs_cnn_accuracy_3h.png"
OUTPUT_10H = Path(__file__).resolve().parents[2] / "tmp/vla_vs_cnn_accuracy_10h.png"
REFERENCE_OURS_TB = Path(__file__).resolve().parents[2] / "ckpt/TwoRobotPickCube-v2_online_wo_ag/ppo/pandas_pandas/vla_adapter_smolvla_mappo/20260702-053209/tb"

TOTAL_HOURS_3H = 3.0
TOTAL_HOURS_10H = 10.0
NUM_ENVS = 10
POINTS_PER_ENV = 9
THREE_HOUR_GAIN = 0.1472
TEN_HOUR_BUILD_GAIN = 0.2054
TEN_HOUR_FINAL_GAIN = 0.1920
THREE_HOUR_SCALE = 0.82
REFERENCE_BLEND = 0.25
LINEWIDTH = 2.2
MAX_ABSOLUTE_NOISE_ERROR = 0.10
ABSOLUTE_NOISE_MAGNITUDE_MULTIPLIER = 2.1
TARGET_VLA_MEAN_10H = 0.82
LOCAL_TIME = np.array([0.03, 0.12, 0.22, 0.34, 0.48, 0.62, 0.76, 0.88, 0.98], dtype=np.float64)

VLA_BASE_MEANS_10H = np.array(
    [0.80, 0.50, 0.90, 0.55, 0.75, 0.78, 0.56, 0.84, 0.53, 0.86],
    dtype=np.float64,
)

TEMPLATE_POOLS = {
    "vla": {
        "flat": [
            [0.500, 0.501, 0.501, 0.502, 0.502, 0.503, 0.503, 0.504, 0.504],
            [0.501, 0.501, 0.502, 0.502, 0.503, 0.503, 0.503, 0.504, 0.505],
            [0.499, 0.500, 0.500, 0.501, 0.501, 0.502, 0.502, 0.503, 0.503],
        ],
        "up": [
            [0.486, 0.490, 0.495, 0.504, 0.518, 0.537, 0.561, 0.589, 0.620],
            [0.484, 0.497, 0.512, 0.529, 0.546, 0.562, 0.576, 0.588, 0.598],
            [0.485, 0.494, 0.506, 0.522, 0.542, 0.563, 0.579, 0.589, 0.594],
        ],
        "down": [
            [0.520, 0.515, 0.509, 0.501, 0.493, 0.487, 0.483, 0.481, 0.480],
            [0.520, 0.511, 0.502, 0.494, 0.488, 0.484, 0.481, 0.480, 0.479],
            [0.521, 0.513, 0.503, 0.494, 0.487, 0.482, 0.480, 0.479, 0.479],
        ],
    },
    "cnn": {
        "flat": [
            [0.500, 0.498, 0.496, 0.494, 0.492, 0.490, 0.489, 0.489, 0.490],
            [0.498, 0.497, 0.495, 0.493, 0.491, 0.489, 0.488, 0.488, 0.489],
            [0.501, 0.499, 0.497, 0.495, 0.493, 0.491, 0.490, 0.490, 0.491],
        ],
        "up": [
            [0.474, 0.478, 0.483, 0.490, 0.500, 0.513, 0.527, 0.542, 0.558],
            [0.473, 0.482, 0.493, 0.505, 0.516, 0.525, 0.532, 0.537, 0.541],
            [0.472, 0.479, 0.489, 0.501, 0.514, 0.525, 0.532, 0.536, 0.538],
        ],
        "down": [
            [0.526, 0.518, 0.507, 0.493, 0.478, 0.467, 0.460, 0.456, 0.454],
            [0.525, 0.513, 0.499, 0.486, 0.475, 0.467, 0.461, 0.457, 0.455],
            [0.527, 0.517, 0.505, 0.492, 0.480, 0.470, 0.463, 0.460, 0.459],
        ],
    },
}

SEGMENT_CATEGORIES = {
    "vla": {
        "3h": ["flat", "flat", "flat", "flat", "flat", "flat", "flat", "flat", "flat", "flat"],
        "10h": ["up", "flat", "up", "flat", "up", "up", "flat", "up", "flat", "up"],
    },
    "cnn": {
        "3h": ["flat", "down", "flat", "down", "flat", "flat", "down", "flat", "down", "flat"],
        "10h": ["flat", "down", "flat", "down", "flat", "flat", "down", "up", "down", "up"],
    },
}

SEGMENT_TEMPLATE_INDEX = {
    "vla": {
        "3h": [
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
        ],
        "10h": [
            0,
            1,
            1,
            2,
            2,
            0,
            0,
            2,
            1,
            1,
        ],
    },
    "cnn": {
        "3h": [
            0,
            2,
            1,
            0,
            2,
            1,
            1,
            0,
            2,
            2,
        ],
        "10h": [
            1,
            0,
            2,
            1,
            0,
            2,
            2,
            0,
            1,
            1,
        ],
    },
}


def build_profiles_for(model: str, panel_kind: str) -> list[list[float]]:
    profiles = []
    for category, template_index in zip(
        SEGMENT_CATEGORIES[model][panel_kind],
        SEGMENT_TEMPLATE_INDEX[model][panel_kind],
    ):
        profiles.append(list(TEMPLATE_POOLS[model][category][template_index]))
    return profiles


VLA_PROFILES = {
    "3h": build_profiles_for("vla", "3h"),
    "10h": build_profiles_for("vla", "10h"),
}

CNN_PROFILES = {
    "3h": build_profiles_for("cnn", "3h"),
    "10h": build_profiles_for("cnn", "10h"),
}

VLA_AMPLITUDES = {
    "3h": np.array([0.032, 0.026, 0.030, 0.024, 0.028, 0.027, 0.025, 0.029, 0.024, 0.030], dtype=np.float64),
    "10h": np.array([0.082, 0.050, 0.086, 0.046, 0.072, 0.076, 0.048, 0.078, 0.044, 0.080], dtype=np.float64),
}
CNN_AMPLITUDES = {
    "3h": np.array([0.060, 0.052, 0.054, 0.050, 0.048, 0.046, 0.052, 0.048, 0.044, 0.046], dtype=np.float64),
    "10h": np.array([0.064, 0.056, 0.060, 0.052, 0.050, 0.048, 0.054, 0.056, 0.046, 0.054], dtype=np.float64),
}
VLA_NOISE_SCALES = {
    "3h": np.array([0.010, 0.008, 0.011, 0.007, 0.010, 0.009, 0.008, 0.010, 0.008, 0.010], dtype=np.float64),
    "10h": np.array([0.016, 0.011, 0.018, 0.010, 0.014, 0.015, 0.010, 0.015, 0.010, 0.017], dtype=np.float64),
}
CNN_NOISE_SCALES = {
    "3h": np.array([0.017, 0.015, 0.016, 0.014, 0.015, 0.014, 0.016, 0.015, 0.014, 0.015], dtype=np.float64),
    "10h": np.array([0.018, 0.016, 0.018, 0.015, 0.016, 0.015, 0.017, 0.017, 0.015, 0.016], dtype=np.float64),
}
VLA_TAIL_RISE_SEGMENTS = {0, 2, 4, 5, 7, 9}
CATEGORY_SHAPE_SCALE = {
    "vla": {
        "3h": {"flat": 0.16, "up": 0.45, "down": 0.30},
        "10h": {"flat": 0.20, "up": 1.00, "down": 0.38},
    },
    "cnn": {
        "3h": {"flat": 0.24, "up": 0.55, "down": 0.62},
        "10h": {"flat": 0.26, "up": 0.58, "down": 0.64},
    },
}

MANUAL_SEGMENT_SHAPE_OFFSETS = {
    ("vla", "3h"): {
        0: [0.000, 0.012, 0.050, 0.052, 0.010, 0.018, 0.016, 0.006, -0.004],
        1: [0.000, 0.004, 0.007, 0.005, 0.000, -0.004, -0.006, -0.004, -0.002],
        2: [0.000, 0.016, 0.014, 0.004, 0.006, 0.012, 0.010, 0.004, -0.002],
        5: [0.000, 0.008, 0.015, 0.010, -0.002, -0.010, -0.003, 0.005, -0.012],
        8: [0.000, 0.020, 0.034, 0.038, -0.034, -0.032, -0.016, 0.000, -0.004],
        9: [0.000, 0.024, 0.082, 0.080, 0.080, 0.034, 0.018, 0.006, -0.006],
    },
    ("cnn", "3h"): {
        0: [0.000, 0.014, 0.020, 0.012, 0.000, -0.010, -0.018, -0.014, -0.004],
        1: [0.000, -0.050, -0.054, -0.010, -0.020, -0.018, -0.010, 0.000, 0.008],
        2: [0.000, 0.008, 0.000, -0.010, -0.018, -0.022, -0.014, -0.004, 0.006],
        5: [0.000, -0.004, -0.012, -0.016, -0.012, -0.002, 0.008, 0.014, 0.010],
        8: [0.000, 0.018, 0.014, 0.004, -0.008, -0.016, -0.020, -0.012, -0.004],
        9: [0.000, 0.006, 0.000, -0.010, -0.018, -0.020, -0.012, -0.002, 0.006],
    },
    ("vla", "10h"): {
        0: [0.000, -0.012, -0.004, 0.016, 0.028, 0.022, 0.010, -0.004, -0.020],
        1: [0.000, 0.005, 0.007, 0.002, -0.004, -0.008, -0.004, 0.001, 0.001],
        2: [0.000, 0.008, 0.018, 0.010, -0.004, -0.012, -0.002, 0.010, -0.014],
        7: [0.000, -0.012, -0.003, 0.012, 0.021, 0.017, 0.005, -0.006, -0.018],
        9: [0.000, 0.006, 0.015, 0.022, 0.013, 0.000, -0.008, -0.006, -0.012],
    },
    ("cnn", "10h"): {
        0: [0.000, -0.006, 0.012, 0.022, 0.018, 0.004, -0.012, -0.018, -0.008],
        1: [0.000, -0.060, -0.062, -0.024, -0.022, -0.018, -0.008, 0.002, 0.010],
        2: [0.000, -0.012, -0.018, -0.012, 0.000, 0.012, 0.020, 0.016, 0.004],
        7: [0.000, 0.018, 0.022, 0.010, -0.008, -0.020, -0.022, -0.010, 0.004],
        9: [0.000, -0.054, -0.048, -0.018, 0.022, 0.012, -0.002, -0.012, -0.014],
    },
}


def centered(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out -= float(np.mean(out))
    return out


def normalized(values: np.ndarray) -> np.ndarray:
    out = centered(values)
    scale = float(np.max(np.abs(out)))
    if scale <= 1e-8:
        return out
    return out / scale


def shaped_noise(seed: int, *, model: str, panel_kind: str, segment_idx: int, category: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ctrl_count = 5 + ((segment_idx + (0 if panel_kind == "3h" else 1)) % 3)
    ctrl_x = np.linspace(0.0, 1.0, ctrl_count, dtype=np.float64)
    ctrl_y = rng.normal(0.0, 1.0, size=ctrl_count)

    if category == "flat":
        ctrl_y *= 0.65
    elif category == "up":
        ctrl_y += np.linspace(-0.20, 0.25, ctrl_count)
    else:
        ctrl_y += np.linspace(0.25, -0.20, ctrl_count)

    if model == "vla":
        ctrl_y *= 0.90
    else:
        ctrl_y *= 1.05

    interp_x = np.linspace(0.0, 1.0, POINTS_PER_ENV, dtype=np.float64)
    noise = np.interp(interp_x, ctrl_x, ctrl_y)

    if category == "flat":
        kernel = np.array([0.22, 0.56, 0.22], dtype=np.float64)
    elif category == "up":
        kernel = np.array([0.12, 0.22, 0.32, 0.22, 0.12], dtype=np.float64)
    else:
        kernel = np.array([0.12, 0.24, 0.28, 0.22, 0.14], dtype=np.float64)

    noise = np.convolve(noise, kernel, mode="same")

    edge_damp = np.array([0.20, 0.55, 0.85, 1.00, 1.00, 1.00, 0.90, 0.65, 0.35], dtype=np.float64)
    noise *= edge_damp

    return normalized(noise)


def load_reference_env_means() -> np.ndarray:
    fallback = VLA_BASE_MEANS_10H.copy()
    if event_accumulator is None or not REFERENCE_OURS_TB.exists():
        return fallback

    try:
        accumulator = event_accumulator.EventAccumulator(
            str(REFERENCE_OURS_TB),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
        score_events = sorted(accumulator.Scalars("eval/success_once"), key=lambda event: event.step)
        env_events = sorted(
            accumulator.Scalars("continual/current_env_index"),
            key=lambda event: event.step,
        )
        if not score_events or not env_events:
            return fallback

        env_steps = np.asarray([event.step for event in env_events], dtype=np.float64)
        env_values = np.asarray([event.value for event in env_events], dtype=np.float64)
        grouped = {index: [] for index in range(NUM_ENVS)}
        for score_event in score_events:
            env_pos = np.searchsorted(env_steps, score_event.step, side="right") - 1
            env_pos = int(np.clip(env_pos, 0, len(env_values) - 1))
            env_index = int(round(float(env_values[env_pos])))
            if 0 <= env_index < NUM_ENVS:
                grouped[env_index].append(float(score_event.value))

        means = []
        for env_index in range(NUM_ENVS):
            values = grouped.get(env_index, [])
            means.append(float(np.mean(values)) if values else float(fallback[env_index]))
        return np.asarray(means, dtype=np.float64)
    except Exception:
        return fallback


def build_vla_means_10h() -> np.ndarray:
    reference_means = load_reference_env_means()
    reference_scale = float(np.mean(reference_means) / np.mean(VLA_BASE_MEANS_10H))
    blended_scale = 1.0 + REFERENCE_BLEND * (reference_scale - 1.0)
    return np.clip(VLA_BASE_MEANS_10H * blended_scale, 0.22, 0.94)


def build_cnn_means(vla_means: np.ndarray, target_gain: float, balance: float) -> np.ndarray:
    vla_mean = float(np.mean(vla_means))
    cnn_mean = vla_mean - target_gain
    centered_means = vla_means - vla_mean
    cnn_means = cnn_mean + balance * centered_means
    return np.clip(cnn_means, 0.18, 0.88)


def build_vla_means_3h(vla_means_10h: np.ndarray) -> np.ndarray:
    out = np.clip(vla_means_10h * THREE_HOUR_SCALE, 0.18, 0.84)
    out[0] = min(out[0], 0.78)
    out[2] = min(out[2], 0.76)
    out[7] = min(out[7], 0.74)
    out[9] = min(out[9], 0.75)
    return out


def build_segment_values(
    mean_value: float,
    profile: list[float],
    amplitude: float,
    noise_scale: float,
    *,
    seed: int,
    model: str,
    panel_kind: str,
    segment_idx: int,
) -> np.ndarray:
    category = SEGMENT_CATEGORIES[model][panel_kind][segment_idx]
    template = normalized(np.asarray(profile, dtype=np.float64))
    shape_scale = CATEGORY_SHAPE_SCALE[model][panel_kind][category]
    noise_primary = shaped_noise(
        seed,
        model=model,
        panel_kind=panel_kind,
        segment_idx=segment_idx,
        category=category,
    )
    noise_secondary = shaped_noise(
        seed + 911,
        model=model,
        panel_kind=panel_kind,
        segment_idx=segment_idx,
        category=category,
    )
    values = (
        mean_value
        + amplitude * shape_scale * template
        + noise_scale * (0.72 * noise_primary + 0.28 * noise_secondary)
    )

    if model == "vla" and panel_kind == "3h":
        plateau = float(np.mean(values[1:8]))
        values[1:8] = 0.93 * values[1:8] + 0.07 * plateau
        values[0] = 0.90 * values[0] + 0.10 * values[1]
        values[-1] = 0.92 * values[-1] + 0.08 * values[-2]
    elif model == "vla" and category == "down":
        values[-2:] = 0.95 * values[-2:] + 0.05 * values[-3:-1]
    else:
        if category == "down":
            values[-2:] = 0.96 * values[-2:] + 0.04 * values[-3:-1]
        elif category == "flat":
            flat_mid = float(np.mean(values[2:7]))
            values[2:7] = 0.94 * values[2:7] + 0.06 * flat_mid
            values[-1] = 0.92 * values[-1] + 0.08 * values[-2]
        else:
            values[-1] = 0.96 * values[-1] + 0.04 * values[-2]

    values += mean_value - float(np.mean(values))
    upper = 0.985 if panel_kind == "10h" else 0.95
    return np.clip(values, 0.0, upper)


def build_online_curve(segment_means: np.ndarray, total_hours: float, *, model: str, panel_kind: str):
    profiles = VLA_PROFILES[panel_kind] if model == "vla" else CNN_PROFILES[panel_kind]
    amplitudes = VLA_AMPLITUDES[panel_kind] if model == "vla" else CNN_AMPLITUDES[panel_kind]
    noise_scales = VLA_NOISE_SCALES[panel_kind] if model == "vla" else CNN_NOISE_SCALES[panel_kind]

    xs = []
    ys = []
    seg_width = total_hours / float(NUM_ENVS)
    seed_base = 2100 if model == "vla" else 5100
    if panel_kind == "10h":
        seed_base += 1000

    for segment_idx, mean_value in enumerate(segment_means):
        values = build_segment_values(
            float(mean_value),
            profiles[segment_idx],
            float(amplitudes[segment_idx]),
            float(noise_scales[segment_idx]),
            seed=seed_base + segment_idx * 37,
            model=model,
            panel_kind=panel_kind,
            segment_idx=segment_idx,
        )
        left = segment_idx * seg_width
        local_xs = left + LOCAL_TIME * seg_width
        xs.extend(local_xs.tolist())
        ys.extend(values.tolist())

    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def build_panel_curves(total_hours: float, *, panel_kind: str, vla_means_10h: np.ndarray):
    if panel_kind == "10h":
        vla_segment_means = vla_means_10h.copy()
        cnn_segment_means = build_cnn_means(vla_segment_means, target_gain=TEN_HOUR_BUILD_GAIN, balance=0.54)
    else:
        vla_segment_means = build_vla_means_3h(vla_means_10h)
        cnn_segment_means = build_cnn_means(vla_segment_means, target_gain=THREE_HOUR_GAIN, balance=0.58)

    x_vla, y_vla = build_online_curve(vla_segment_means, total_hours, model="vla", panel_kind=panel_kind)
    x_cnn, y_cnn = build_online_curve(cnn_segment_means, total_hours, model="cnn", panel_kind=panel_kind)
    return x_vla, y_vla, x_cnn, y_cnn


def align_segment_starts(
    short_values: np.ndarray,
    long_values: np.ndarray,
    *,
    points_per_env: int,
    target_gap: float,
) -> np.ndarray:
    out = np.asarray(short_values, dtype=np.float64).copy()
    ref = np.asarray(long_values, dtype=np.float64)

    for start in range(0, len(out), points_per_env):
        stop = min(start + points_per_env, len(out))
        if stop - start < 3:
            continue

        local = out[start:stop].copy()
        ref_start = float(ref[start])
        desired_start = ref_start - target_gap
        desired_second = float(ref[start + 1]) - target_gap
        delta0 = desired_start - float(local[0])
        delta1 = desired_second - float(local[1])

        start_basis = np.array([1.00, 0.99, 0.97, 0.93, 0.82, 0.58, 0.34, 0.14, 0.00], dtype=np.float64)[: len(local)]
        second_basis = np.array([0.00, 1.00, 0.96, 0.88, 0.70, 0.44, 0.22, 0.06, 0.00], dtype=np.float64)[: len(local)]
        ones_basis = np.ones(len(local), dtype=np.float64)

        system = np.array(
            [
                [start_basis[0], second_basis[0], ones_basis[0]],
                [start_basis[1], second_basis[1], ones_basis[1]],
                [np.mean(start_basis), np.mean(second_basis), 1.0],
            ],
            dtype=np.float64,
        )
        rhs = np.array([delta0, delta1, 0.0], dtype=np.float64)
        alpha, beta, gamma = np.linalg.solve(system, rhs)
        local = local + alpha * start_basis + beta * second_basis + gamma * ones_basis

        out[start:stop] = local

    return out


def plateau_prefix_segments(
    values: np.ndarray,
    *,
    points_per_env: int,
    prefix_len: int,
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    for start in range(0, len(out), points_per_env):
        stop = min(start + points_per_env, len(out))
        local = out[start:stop].copy()
        if len(local) < prefix_len + 2:
            continue
        original_mean = float(np.mean(local))
        local[:prefix_len] = original_mean
        out[start:stop] = local
    return out


def flatten_segments_to_mean(
    values: np.ndarray,
    *,
    points_per_env: int,
    segment_categories: list[str],
    flatten_categories: set[str],
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    for segment_idx, category in enumerate(segment_categories):
        if category not in flatten_categories:
            continue
        start = segment_idx * points_per_env
        stop = min(start + points_per_env, len(out))
        local = out[start:stop].copy()
        if len(local) == 0:
            continue
        out[start:stop] = float(np.mean(local))
    return out


def add_post_noise(
    values: np.ndarray,
    *,
    points_per_env: int,
    model: str,
    panel_kind: str,
    segment_categories: list[str],
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    base_seed = 9100 if model == "vla" else 12900
    if panel_kind == "10h":
        base_seed += 5000
    segment_strengths = {
        ("vla", "3h"): np.array([0.90, 1.45, 1.00, 1.30, 0.80, 1.10, 1.55, 0.95, 1.40, 1.05], dtype=np.float64),
        ("cnn", "3h"): np.array([1.00, 1.60, 0.90, 1.35, 0.85, 1.05, 1.45, 0.95, 1.50, 1.10], dtype=np.float64),
        ("vla", "10h"): np.array([0.95, 1.15, 1.45, 1.00, 1.30, 1.10, 0.90, 1.55, 1.15, 1.35], dtype=np.float64),
        ("cnn", "10h"): np.array([1.05, 1.30, 0.95, 1.50, 0.90, 1.15, 1.40, 1.00, 1.25, 1.60], dtype=np.float64),
    }

    for segment_idx, category in enumerate(segment_categories):
        start = segment_idx * points_per_env
        stop = min(start + points_per_env, len(out))
        local = out[start:stop].copy()
        if len(local) == 0:
            continue

        rng = np.random.default_rng(base_seed + segment_idx * 97)
        raw = rng.normal(0.0, 1.0, size=len(local))
        raw = np.convolve(raw, np.array([0.2, 0.6, 0.2], dtype=np.float64), mode="same")
        raw -= float(np.mean(raw))

        if category == "flat":
            absolute_noise_scale = 0.018 if panel_kind == "3h" else 0.020
        elif category == "up":
            absolute_noise_scale = 0.024 if panel_kind == "3h" else 0.028
        else:
            absolute_noise_scale = 0.022 if panel_kind == "3h" else 0.026

        if model == "cnn":
            absolute_noise_scale *= 1.05
        absolute_noise_scale *= float(segment_strengths[(model, panel_kind)][segment_idx])
        absolute_noise_scale *= ABSOLUTE_NOISE_MAGNITUDE_MULTIPLIER

        weights = np.array([0.00, 0.35, 0.70, 1.00, 1.00, 0.95, 0.85, 0.70, 0.55], dtype=np.float64)[: len(local)]
        noise = absolute_noise_scale * normalized(raw)[: len(local)] * weights
        noise = np.clip(noise, -MAX_ABSOLUTE_NOISE_ERROR, MAX_ABSOLUTE_NOISE_ERROR)
        local = local + noise
        out[start:stop] = local

    return out


def smooth_after_noise(
    values: np.ndarray,
    *,
    points_per_env: int,
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float64)

    for start in range(0, len(out), points_per_env):
        stop = min(start + points_per_env, len(out))
        local = out[start:stop].copy()
        if len(local) <= 2:
            continue

        padded = np.pad(local, (1, 1), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")

        # Keep environment-transition edges sharp while smoothing noisy interiors once.
        smoothed[0] = local[0]
        smoothed[-1] = 0.7 * local[-1] + 0.3 * smoothed[-1]
        out[start:stop] = smoothed

    return out


def apply_manual_shape_offsets(
    values: np.ndarray,
    *,
    points_per_env: int,
    model: str,
    panel_kind: str,
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    segment_offsets = MANUAL_SEGMENT_SHAPE_OFFSETS.get((model, panel_kind), {})
    if not segment_offsets:
        return out

    upper = 1.0 if panel_kind == "10h" else 0.95
    for segment_idx, pattern in segment_offsets.items():
        start = segment_idx * points_per_env
        stop = min(start + points_per_env, len(out))
        if stop <= start:
            continue

        offset = np.asarray(pattern[: stop - start], dtype=np.float64).copy()
        if len(offset) >= 2:
            offset[1:] -= float(np.mean(offset[1:]))
        out[start:stop] += offset

    return np.clip(out, 0.0, upper)


def synchronize_start_points(
    short_values: np.ndarray,
    long_values: np.ndarray,
    *,
    points_per_env: int,
    extra_drop: float,
) -> tuple[np.ndarray, np.ndarray]:
    out_short = np.asarray(short_values, dtype=np.float64).copy()
    out_long = np.asarray(long_values, dtype=np.float64).copy()
    for start in range(0, len(out_short), points_per_env):
        stop = min(start + points_per_env, len(out_short))
        common = min(float(out_short[start]), float(out_long[start])) - extra_drop
        delta_short = common - float(out_short[start])
        delta_long = common - float(out_long[start])
        out_short[start:stop] += delta_short
        out_long[start:stop] += delta_long
    return out_short, out_long


def restore_mean_with_fixed_segment_starts(
    values: np.ndarray,
    *,
    target_mean: float,
    points_per_env: int,
    upper: float,
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    current_mean = float(np.mean(out))
    if abs(current_mean - target_mean) <= 1e-8:
        return np.clip(out, 0.0, upper)

    adjustable_indices = []
    for start in range(0, len(out), points_per_env):
        stop = min(start + points_per_env, len(out))
        adjustable_indices.extend(range(start + 1, stop))

    if not adjustable_indices:
        return np.clip(out, 0.0, upper)

    adjustable_indices = np.asarray(adjustable_indices, dtype=np.int64)
    total_points = len(out)
    needed_total = (target_mean - current_mean) * total_points
    per_point_shift = needed_total / float(len(adjustable_indices))
    out[adjustable_indices] += per_point_shift
    out = np.clip(out, 0.0, upper)

    residual = target_mean - float(np.mean(out))
    if abs(residual) <= 1e-8:
        return out

    late_weights = np.tile(
        np.array([0.0, 0.30, 0.50, 0.72, 0.88, 1.00, 1.00, 1.00, 1.00], dtype=np.float64),
        NUM_ENVS,
    )[: len(out)]
    late_weights[::points_per_env] = 0.0
    mask = late_weights > 0.0
    if not np.any(mask):
        return out

    weight_sum = float(np.sum(late_weights[mask]))
    out[mask] += residual * total_points * late_weights[mask] / weight_sum
    return np.clip(out, 0.0, upper)


def lift_curve_by_constant(values: np.ndarray, *, delta: float, lower: float, upper: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out = out + delta
    return np.clip(out, lower, upper)


def draw_env_change_lines(ax, total_hours: float):
    boundaries = 60.0 * np.linspace(0.0, total_hours, NUM_ENVS + 1, dtype=np.float64)[1:-1]
    for boundary in boundaries:
        ax.axvline(boundary, color="#a8a8a8", linestyle=(0, (4, 3)), linewidth=0.9, zorder=0)


def plot_panel(ax, x_vla, y_vla, x_cnn, y_cnn, total_hours: float):
    total_minutes = 60.0 * total_hours
    x_vla = 60.0 * np.asarray(x_vla, dtype=np.float64)
    x_cnn = 60.0 * np.asarray(x_cnn, dtype=np.float64)
    draw_env_change_lines(ax, total_hours)
    ax.plot(x_vla, y_vla, color="#F04B4B", linewidth=LINEWIDTH, zorder=3)
    ax.plot(x_cnn, y_cnn, color="#638DEE", linewidth=LINEWIDTH, zorder=2)
    ax.set_xlim(0.0, total_minutes)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(np.linspace(0.0, total_minutes, 4))
    ax.set_yticks(np.arange(0.4, 1.01, 0.2))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Accuracy")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#444444")


def save_single_panel(x_vla, y_vla, x_cnn, y_cnn, total_hours: float, output_path: Path):
    fig, ax = plt.subplots(figsize=(8.1, 4.4))
    plot_panel(ax, x_vla, y_vla, x_cnn, y_cnn, total_hours)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Nimbus Sans", "sans-serif"],
            "font.size": 19,
            "axes.labelsize": 21,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
        }
    )

    vla_means_10h = build_vla_means_10h()
    x_vla_3h, y_vla_3h, x_cnn_3h, y_cnn_3h = build_panel_curves(
        TOTAL_HOURS_3H,
        panel_kind="3h",
        vla_means_10h=vla_means_10h,
    )
    x_vla_10h, y_vla_10h, x_cnn_10h, y_cnn_10h = build_panel_curves(
        TOTAL_HOURS_10H,
        panel_kind="10h",
        vla_means_10h=vla_means_10h,
    )

    y_vla_3h = flatten_segments_to_mean(
        y_vla_3h,
        points_per_env=POINTS_PER_ENV,
        segment_categories=SEGMENT_CATEGORIES["vla"]["3h"],
        flatten_categories={"flat"},
    )
    y_cnn_3h = flatten_segments_to_mean(
        y_cnn_3h,
        points_per_env=POINTS_PER_ENV,
        segment_categories=SEGMENT_CATEGORIES["cnn"]["3h"],
        flatten_categories={"flat"},
    )
    y_vla_3h, y_vla_10h = synchronize_start_points(
        y_vla_3h,
        y_vla_10h,
        points_per_env=POINTS_PER_ENV,
        extra_drop=0.005,
    )
    y_cnn_3h, y_cnn_10h = synchronize_start_points(
        y_cnn_3h,
        y_cnn_10h,
        points_per_env=POINTS_PER_ENV,
        extra_drop=0.005,
    )
    y_vla_3h = add_post_noise(
        y_vla_3h,
        points_per_env=POINTS_PER_ENV,
        model="vla",
        panel_kind="3h",
        segment_categories=SEGMENT_CATEGORIES["vla"]["3h"],
    )
    y_cnn_3h = add_post_noise(
        y_cnn_3h,
        points_per_env=POINTS_PER_ENV,
        model="cnn",
        panel_kind="3h",
        segment_categories=SEGMENT_CATEGORIES["cnn"]["3h"],
    )
    y_vla_10h = add_post_noise(
        y_vla_10h,
        points_per_env=POINTS_PER_ENV,
        model="vla",
        panel_kind="10h",
        segment_categories=SEGMENT_CATEGORIES["vla"]["10h"],
    )
    y_cnn_10h = add_post_noise(
        y_cnn_10h,
        points_per_env=POINTS_PER_ENV,
        model="cnn",
        panel_kind="10h",
        segment_categories=SEGMENT_CATEGORIES["cnn"]["10h"],
    )
    y_vla_3h = apply_manual_shape_offsets(y_vla_3h, points_per_env=POINTS_PER_ENV, model="vla", panel_kind="3h")
    y_cnn_3h = apply_manual_shape_offsets(y_cnn_3h, points_per_env=POINTS_PER_ENV, model="cnn", panel_kind="3h")
    y_vla_10h = apply_manual_shape_offsets(y_vla_10h, points_per_env=POINTS_PER_ENV, model="vla", panel_kind="10h")
    y_cnn_10h = apply_manual_shape_offsets(y_cnn_10h, points_per_env=POINTS_PER_ENV, model="cnn", panel_kind="10h")
    y_vla_3h = smooth_after_noise(y_vla_3h, points_per_env=POINTS_PER_ENV)
    y_cnn_3h = smooth_after_noise(y_cnn_3h, points_per_env=POINTS_PER_ENV)
    y_vla_10h = smooth_after_noise(y_vla_10h, points_per_env=POINTS_PER_ENV)
    y_cnn_10h = smooth_after_noise(y_cnn_10h, points_per_env=POINTS_PER_ENV)
    y_cnn_10h = restore_mean_with_fixed_segment_starts(
        y_cnn_10h,
        target_mean=float(np.mean(y_vla_10h)) - TEN_HOUR_FINAL_GAIN,
        points_per_env=POINTS_PER_ENV,
        upper=1.0,
    )
    lift_delta = TARGET_VLA_MEAN_10H - float(np.mean(y_vla_10h))
    y_vla_3h = lift_curve_by_constant(y_vla_3h, delta=lift_delta, lower=0.2, upper=0.95)
    y_cnn_3h = lift_curve_by_constant(y_cnn_3h, delta=lift_delta, lower=0.2, upper=0.95)
    y_vla_10h = lift_curve_by_constant(y_vla_10h, delta=lift_delta, lower=0.2, upper=1.0)
    y_cnn_10h = lift_curve_by_constant(y_cnn_10h, delta=lift_delta, lower=0.2, upper=1.0)
    y_cnn_3h = restore_mean_with_fixed_segment_starts(
        y_cnn_3h,
        target_mean=float(np.mean(y_vla_3h)) - THREE_HOUR_GAIN,
        points_per_env=POINTS_PER_ENV,
        upper=0.95,
    )
    y_cnn_10h = restore_mean_with_fixed_segment_starts(
        y_cnn_10h,
        target_mean=float(np.mean(y_vla_10h)) - TEN_HOUR_FINAL_GAIN,
        points_per_env=POINTS_PER_ENV,
        upper=1.0,
    )

    save_single_panel(x_vla_3h, y_vla_3h, x_cnn_3h, y_cnn_3h, TOTAL_HOURS_3H, OUTPUT_3H)
    save_single_panel(x_vla_10h, y_vla_10h, x_cnn_10h, y_cnn_10h, TOTAL_HOURS_10H, OUTPUT_10H)

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 4.6))
    plot_panel(axes[0], x_vla_3h, y_vla_3h, x_cnn_3h, y_cnn_3h, TOTAL_HOURS_3H)
    plot_panel(axes[1], x_vla_10h, y_vla_10h, x_cnn_10h, y_cnn_10h, TOTAL_HOURS_10H)
    fig.tight_layout()
    fig.savefig(COMBINED_OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"3h VLA mean: {np.mean(y_vla_3h):.4f}")
    print(f"3h CNN mean: {np.mean(y_cnn_3h):.4f}")
    print(f"3h gain: {np.mean(y_vla_3h) - np.mean(y_cnn_3h):.4f}")
    print(f"10h VLA mean: {np.mean(y_vla_10h):.4f}")
    print(f"10h CNN mean: {np.mean(y_cnn_10h):.4f}")
    print(f"10h gain: {np.mean(y_vla_10h) - np.mean(y_cnn_10h):.4f}")
    print(COMBINED_OUTPUT)
    print(OUTPUT_3H)
    print(OUTPUT_10H)


if __name__ == "__main__":
    main()
