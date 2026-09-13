import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import transformers
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.utils.hub import cached_file

from train.reinforcement_learning.utils import RunningMeanStd


def configure_hf_mirror() -> None:
    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if endpoint:
        return
    use_mirror = os.environ.get("USE_HF_MIRROR", "0").strip().lower()
    if use_mirror in {"1", "true", "yes", "on"}:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


configure_hf_mirror()


class FallbackImageProcessor:
    def __init__(
        self,
        image_size: int = 112,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ) -> None:
        self.image_size = int(image_size)
        self.mean = torch.tensor(mean or [0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std or [0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - self.mean) / self.std

    def __call__(self, images: Sequence[Image.Image], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError(f"Unsupported return_tensors={return_tensors}, expected 'pt'")
        pixel_values = torch.stack([self._process_image(image) for image in images], dim=0)
        return {"pixel_values": pixel_values}


class ForwardKwargFilterWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        signature = inspect.signature(model.forward)
        self.allowed_kwargs = set(signature.parameters.keys())
        self.config = getattr(model, "config", None)

    def forward(self, *args, **kwargs):
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in self.allowed_kwargs}
        return self.model(*args, **filtered_kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def set_module_attr_recursive(module: nn.Module, attr_name: str, value: Any, max_depth: int = 6) -> None:
    visited = set()

    def _set(obj: Any, depth: int) -> None:
        if obj is None or depth > max_depth:
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        try:
            setattr(obj, attr_name, value)
        except Exception:
            pass

        for child_name in ["model", "base_model", "module"]:
            child = getattr(obj, child_name, None)
            if child is not None and child is not obj:
                _set(child, depth + 1)

    _set(module, 0)


TASK_PROMPT = (
    "coordinate two robot arms to pick up the red cube and move it to the target goal position."
)
_DEBUG_RGB_SAVED = False
VLA_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv",
    "proj",
    "fc1",
    "fc2",
    "fc3",
    "q",
    "kv",
]


def build_agent_role_prompt(agent_name: str) -> str:
    if agent_name.endswith("-0"):
        role = "the left robot arm"
        partner = "the right robot arm"
    else:
        role = "the right robot arm"
        partner = "the left robot arm"
    return (
        f"act as {role}; coordinate with {partner} to pick up the red cube "
        "and move it to the target goal position."
    )


def resolve_attention_implementation(requested: str) -> str:
    valid = {"eager", "sdpa", "flash_attention_2"}
    if requested not in valid:
        raise ValueError(f"Unsupported attention implementation: {requested}. Expected one of {sorted(valid)}")
    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print("[setup] flash_attn is not installed; falling back to SDPA attention")
        return "sdpa"
    return requested


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


def maybe_save_debug_rgb(obs: Dict[str, Any], output_path: Union[str, Path] = "debug_rgb.png") -> None:
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


def extract_agent_state_from_obs(obs: Dict[str, Any], agent_name: str) -> torch.Tensor:
    is_left = agent_name.endswith("-0")
    tcp_key = "left_arm_tcp" if is_left else "right_arm_tcp"
    tcp_to_cube_key = "left_arm_tcp_to_cube_pos" if is_left else "right_arm_tcp_to_cube_pos"

    qpos = _ensure_2d(_as_tensor(obs["agent"][agent_name]["qpos"], dtype=torch.float32))
    batch_size = qpos.shape[0]
    qvel = _ensure_2d(_as_tensor(obs["agent"][agent_name]["qvel"], dtype=torch.float32))
    tcp = _ensure_2d(_as_tensor(obs["extra"][tcp_key], dtype=torch.float32))
    tcp_to_cube = _ensure_2d(_as_tensor(obs["extra"][tcp_to_cube_key], dtype=torch.float32))
    cube_pose = _ensure_2d(_as_tensor(obs["extra"]["cube_pose"], dtype=torch.float32))
    cube_to_goal = _ensure_2d(_as_tensor(obs["extra"]["cube_to_goal_pos"], dtype=torch.float32))
    stage = _ensure_2d(_as_tensor(obs["extra"]["stage"], dtype=torch.float32), batch_size=batch_size)

    return torch.cat(
        [qpos, qvel, tcp, tcp_to_cube, cube_pose, cube_to_goal, stage],
        dim=-1,
    )


def extract_global_state_from_obs(obs: Dict[str, Any], agent_names: List[str]) -> torch.Tensor:
    left_name, right_name = agent_names
    left_qpos = _ensure_2d(_as_tensor(obs["agent"][left_name]["qpos"], dtype=torch.float32))
    batch_size = left_qpos.shape[0]
    left_qvel = _ensure_2d(_as_tensor(obs["agent"][left_name]["qvel"], dtype=torch.float32))
    right_qpos = _ensure_2d(_as_tensor(obs["agent"][right_name]["qpos"], dtype=torch.float32))
    right_qvel = _ensure_2d(_as_tensor(obs["agent"][right_name]["qvel"], dtype=torch.float32))
    left_tcp = _ensure_2d(_as_tensor(obs["extra"]["left_arm_tcp"], dtype=torch.float32))
    right_tcp = _ensure_2d(_as_tensor(obs["extra"]["right_arm_tcp"], dtype=torch.float32))
    cube_pose = _ensure_2d(_as_tensor(obs["extra"]["cube_pose"], dtype=torch.float32))
    left_tcp_to_cube = _ensure_2d(_as_tensor(obs["extra"]["left_arm_tcp_to_cube_pos"], dtype=torch.float32))
    right_tcp_to_cube = _ensure_2d(_as_tensor(obs["extra"]["right_arm_tcp_to_cube_pos"], dtype=torch.float32))
    cube_to_goal = _ensure_2d(_as_tensor(obs["extra"]["cube_to_goal_pos"], dtype=torch.float32))
    stage = _ensure_2d(_as_tensor(obs["extra"]["stage"], dtype=torch.float32), batch_size=batch_size)

    return torch.cat(
        [
            left_qpos,
            left_qvel,
            right_qpos,
            right_qvel,
            left_tcp,
            right_tcp,
            cube_pose,
            left_tcp_to_cube,
            right_tcp_to_cube,
            cube_to_goal,
            stage,
        ],
        dim=-1,
    )


def build_batch_from_obs(obs: Dict[str, Any], agent_names: List[str]) -> Dict[str, Any]:
    maybe_save_debug_rgb(obs)
    batch = {
        "rgb": extract_rgb_batch_from_obs(obs),
        "global_state": extract_global_state_from_obs(obs, agent_names),
    }
    for name in agent_names:
        batch[f"agent_states_{name}"] = extract_agent_state_from_obs(obs, name)
    return batch


class MLPProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualDiscreteActorHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, num_bins: int):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )
        self.logit_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bins),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(
        self,
        action_features: torch.Tensor,
        state_feature: torch.Tensor,
        context_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = action_features.shape
        action_features = self.context_encoder(action_features.reshape(batch_size, seq_len, hidden_dim))
        expanded_state = state_feature.unsqueeze(1).expand(-1, seq_len, -1)
        expanded_context = context_feature if context_feature.ndim == 3 else context_feature.unsqueeze(1)
        expanded_context = expanded_context.expand(-1, seq_len, -1)
        fused = torch.cat([action_features, expanded_state, expanded_context], dim=-1)
        return self.logit_head(fused) * self.residual_scale


class SharedAutoregressiveActionHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, num_bins: int):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )
        self.token_embedding = nn.Embedding(num_bins, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(action_dim, hidden_dim, dtype=torch.float32) * 0.02)
        self.memory_type_embedding = nn.Parameter(torch.randn(3, hidden_dim, dtype=torch.float32) * 0.02)
        self.bos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim, dtype=torch.float32) * 0.02)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bins),
        )
        self.logit_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def build_memory(
        self,
        prompt_hidden: torch.Tensor,
        context_feature: torch.Tensor,
        state_feature: torch.Tensor,
    ) -> torch.Tensor:
        memory = torch.stack([prompt_hidden, context_feature, state_feature], dim=1)
        return memory + self.memory_type_embedding.unsqueeze(0)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32),
            diagonal=1,
        )

    def decode_teacher_forced(self, memory: torch.Tensor, action_bins: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = action_bins.shape
        if seq_len != self.action_dim:
            raise ValueError(f"Expected action sequence length {self.action_dim}, got {seq_len}")
        if seq_len == 1:
            prev_token_embeds = self.bos_embedding.expand(batch_size, 1, -1)
        else:
            prev_tokens = action_bins[:, :-1]
            prev_token_embeds = torch.cat(
                [
                    self.bos_embedding.expand(batch_size, 1, -1),
                    self.token_embedding(prev_tokens),
                ],
                dim=1,
            )
        prev_token_embeds = prev_token_embeds + self.position_embedding[:seq_len].unsqueeze(0)
        decoded = self.decoder(
            tgt=prev_token_embeds,
            memory=memory,
            tgt_mask=self._causal_mask(seq_len, prev_token_embeds.device),
        )
        return self.output_head(decoded) * self.logit_scale

    def sample(
        self,
        memory: torch.Tensor,
        *,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = memory.shape[0]
        sampled_bins = []
        step_log_probs = []
        step_entropies = []

        for step in range(self.action_dim):
            if step == 0:
                prev_embeds = self.bos_embedding.expand(batch_size, 1, -1)
            else:
                prev_tokens = torch.stack(sampled_bins, dim=1)
                prev_embeds = torch.cat(
                    [
                        self.bos_embedding.expand(batch_size, 1, -1),
                        self.token_embedding(prev_tokens),
                    ],
                    dim=1,
                )
            seq_len = prev_embeds.shape[1]
            prev_embeds = prev_embeds + self.position_embedding[:seq_len].unsqueeze(0)
            decoded = self.decoder(
                tgt=prev_embeds,
                memory=memory,
                tgt_mask=self._causal_mask(seq_len, prev_embeds.device),
            )
            logits = self.output_head(decoded[:, -1:, :]).squeeze(1) * self.logit_scale
            categorical = torch.distributions.Categorical(logits=logits)
            next_bin = logits.argmax(dim=-1) if deterministic else categorical.sample()
            sampled_bins.append(next_bin)
            step_log_probs.append(categorical.log_prob(next_bin))
            step_entropies.append(categorical.entropy())

        selected_bins = torch.stack(sampled_bins, dim=1)
        log_prob = torch.stack(step_log_probs, dim=1).sum(dim=1)
        entropy = torch.stack(step_entropies, dim=1).mean(dim=1)
        return selected_bins, log_prob, entropy


def _resolve_local_image_size_from_config(model_name_or_path: str) -> int:
    try:
        config_path = cached_file(model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        force_image_size = config_dict.get("force_image_size")
        if isinstance(force_image_size, int) and force_image_size > 0:
            return force_image_size
        vision_config = config_dict.get("vision_config", {})
        image_size = vision_config.get("image_size")
        if isinstance(image_size, int) and image_size > 0:
            return image_size
    except Exception:
        pass
    return 112


def resolve_vla_image_size(model_name_or_path: str, requested_image_size: Optional[int]) -> int:
    if requested_image_size is not None:
        image_size = int(requested_image_size)
    else:
        image_size = int(_resolve_local_image_size_from_config(model_name_or_path))
    if image_size <= 0:
        raise ValueError(f"Invalid image_size={image_size}, expected a positive integer")
    if image_size % 14 != 0:
        raise ValueError(
            f"Invalid image_size={image_size}. InternVL vision patch_size is 14, so image_size must be divisible by 14."
        )
    patch_grid = image_size // 14
    if patch_grid % 2 != 0:
        raise ValueError(
            f"Invalid image_size={image_size}. InternVL3.5 uses downsample_ratio=0.5 pixel_shuffle, "
            f"so the patch grid size image_size/14={patch_grid} must also be even."
        )
    return image_size


def _load_processor_components(model_name_or_path: str, image_size: Optional[int] = None):
    processor = None
    tokenizer = None
    image_processor = None
    resolved_image_size = resolve_vla_image_size(model_name_or_path, image_size)

    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception:
        processor = None

    if processor is not None:
        tokenizer = getattr(processor, "tokenizer", None)
        image_processor = getattr(processor, "image_processor", None)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if image_processor is None:
        try:
            image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        except Exception:
            image_processor = FallbackImageProcessor(
                image_size=resolved_image_size
            )
    elif hasattr(image_processor, "size"):
        size_cfg = getattr(image_processor, "size")
        if isinstance(size_cfg, dict):
            if "height" in size_cfg:
                size_cfg["height"] = resolved_image_size
            if "width" in size_cfg:
                size_cfg["width"] = resolved_image_size
            if "shortest_edge" in size_cfg:
                size_cfg["shortest_edge"] = resolved_image_size
        elif isinstance(size_cfg, int):
            setattr(image_processor, "size", resolved_image_size)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token

    return processor, tokenizer, image_processor


def _resolve_hidden_dim(config: Any, model: nn.Module) -> int:
    candidates = [
        getattr(config, "hidden_size", None),
        getattr(getattr(config, "text_config", None), "hidden_size", None),
        getattr(getattr(config, "llm_config", None), "hidden_size", None),
        getattr(getattr(config, "language_config", None), "hidden_size", None),
        getattr(getattr(config, "vision_config", None), "hidden_size", None),
        getattr(getattr(model, "config", None), "hidden_size", None),
        getattr(getattr(getattr(model, "language_model", None), "config", None), "hidden_size", None),
        getattr(getattr(getattr(model, "llm", None), "config", None), "hidden_size", None),
    ]
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    return 1024


def _validate_transformers_support(model_name_or_path: str) -> None:
    try:
        config_path = cached_file(model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
    except Exception:
        return

    llm_config = config_dict.get("llm_config", {})
    architectures = llm_config.get("architectures", [])
    architecture = architectures[0] if architectures else None
    if architecture == "Qwen3ForCausalLM" and not hasattr(transformers, "Qwen3Config"):
        raise ImportError(
            "OpenGVLab/InternVL3_5-1B-Instruct requires Qwen3 support in transformers, "
            f"but installed transformers=={transformers.__version__} does not provide Qwen3Config. "
            "Upgrade transformers to >=4.51.1, or switch to a model whose llm_config does not use Qwen3."
        )
    if architecture == "Qwen3MoeForCausalLM" and not hasattr(transformers, "Qwen3MoeConfig"):
        raise ImportError(
            "This model requires Qwen3 MoE support in transformers, "
            f"but installed transformers=={transformers.__version__} does not provide Qwen3MoeConfig. "
            "Upgrade transformers to a version with Qwen3 MoE support."
        )


def _load_backbone(
    model_name_or_path: str,
    *,
    attention_implementation: str,
) -> Tuple[Any, nn.Module]:
    _validate_transformers_support(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    use_flash_attention = attention_implementation == "flash_attention_2" and torch.cuda.is_available()
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None and not use_flash_attention:
        if hasattr(vision_config, "use_flash_attn"):
            vision_config.use_flash_attn = False
        if hasattr(vision_config, "use_fa3"):
            vision_config.use_fa3 = False

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "config": config,
    }
    if attention_implementation:
        load_kwargs["attn_implementation"] = resolve_attention_implementation(attention_implementation)

    errors = []
    for loader in (AutoModel, AutoModelForCausalLM):
        for kwargs in (load_kwargs, {k: v for k, v in load_kwargs.items() if k != "attn_implementation"}):
            try:
                model = loader.from_pretrained(model_name_or_path, **kwargs)
                return config, model
            except Exception as exc:
                errors.append(f"{loader.__name__}: {exc}")

    joined = "\n".join(errors[-4:])
    raise RuntimeError(f"Failed to load InternVL backbone from {model_name_or_path}.\n{joined}")


class SharedInternVLActor(nn.Module):
    def __init__(
        self,
        model_dir: Union[str, Path],
        state_dim: int,
        env_action_dim: int = 4,
        num_agents: int = 2,
        prompt: str = TASK_PROMPT,
        attention_implementation: str = "sdpa",
        image_size: Optional[int] = 112,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        num_action_bins: int = 256,
    ):
        super().__init__()
        self.model_dir = str(model_dir)
        self.state_dim = state_dim
        self.env_action_dim = env_action_dim
        self.num_agents = num_agents
        self.use_lora = use_lora
        self.num_action_bins = num_action_bins
        self.image_size = resolve_vla_image_size(self.model_dir, image_size)
        self.base_prompt = f"What action should the robot take to {prompt}"
        self.img_start_token = "<img>"
        self.img_end_token = "</img>"
        self.img_context_token = "<IMG_CONTEXT>"

        self.processor, self.tokenizer, self.image_processor = _load_processor_components(
            self.model_dir,
            image_size=self.image_size,
        )
        self.config, self.vla = _load_backbone(
            self.model_dir,
            attention_implementation=attention_implementation,
        )
        if self.use_lora:
            self.vla = self._apply_lora(
                self.vla,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        self.hidden_dim = _resolve_hidden_dim(self.config, self.vla)
        self.num_image_token = int((self.image_size // self.config.vision_config.patch_size) ** 2 * (self.config.downsample_ratio ** 2))
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.img_context_token)
        if self.img_context_token_id is None or self.img_context_token_id < 0:
            raise RuntimeError(f"Failed to resolve {self.img_context_token} token id for {self.model_dir}")
        set_module_attr_recursive(self.vla, "img_context_token_id", self.img_context_token_id)

        self.register_buffer(
            "action_bin_centers",
            torch.linspace(-1.0, 1.0, steps=self.num_action_bins, dtype=torch.float32),
            persistent=False,
        )
        self.state_projector = MLPProjector(
            input_dim=state_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        ).to(dtype=torch.float32)
        self.state_role_embeddings = nn.Embedding(self.num_agents, self.hidden_dim)
        self.agent_marker_embeddings = nn.Embedding(self.num_agents, self.hidden_dim)
        self.action_token_embedding = nn.Embedding(self.num_action_bins, self.hidden_dim)
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_action_bins),
        ).to(dtype=torch.float32)
        self.action_logit_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

        self.eval_micro_batch_size = 32
        self._vla_trainable = True

    @property
    def device(self) -> torch.device:
        return next(self.vla.parameters()).device

    @staticmethod
    def _apply_lora(vla: nn.Module, r: int, alpha: int, dropout: float) -> nn.Module:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("`peft` is required to enable VLA LoRA") from exc

        wrapped_vla = ForwardKwargFilterWrapper(vla)
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=VLA_LORA_TARGET_MODULES,
        )
        wrapped = get_peft_model(wrapped_vla, lora_cfg)
        wrapped.print_trainable_parameters()
        return wrapped

    def configure_trainable_modules(self, train_backbone: bool) -> None:
        self._vla_trainable = train_backbone
        if self.use_lora:
            for parameter in self.vla.parameters():
                parameter.requires_grad = False
            for name, parameter in self.vla.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad = train_backbone
        else:
            for parameter in self.vla.parameters():
                parameter.requires_grad = train_backbone
        for module in [
            self.state_projector,
            self.state_role_embeddings,
            self.agent_marker_embeddings,
            self.action_token_embedding,
            self.action_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    @staticmethod
    def _prepare_image(rgb: Union[np.ndarray, torch.Tensor]) -> Image.Image:
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
        return Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert("RGB")

    def _prepare_policy_inputs(
        self,
        rgbs: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if isinstance(rgbs, torch.Tensor):
            rgb_batch = rgbs[..., :3].detach()
            if rgb_batch.device.type != "cpu" or rgb_batch.dtype != torch.uint8 or not rgb_batch.is_contiguous():
                rgb_batch = rgb_batch.to(device="cpu", dtype=torch.uint8).contiguous()
        else:
            rgb_batch = torch.from_numpy(np.asarray(rgbs)[..., :3].astype(np.uint8, copy=False)).contiguous()

        images = [self._prepare_image(rgb) for rgb in rgb_batch]
        batch_size = len(images)
        image_inputs = self.image_processor(images=images, return_tensors="pt")
        questions = [self.base_prompt] * batch_size

        image_tokens = self.img_start_token + self.img_context_token * self.num_image_token + self.img_end_token
        prompt_texts = [f"{image_tokens}\n{question}" for question in questions]
        tokenized = self.tokenizer(prompt_texts, return_tensors="pt", padding=True)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        model_inputs = {
            "input_ids": input_ids.to(self.device, non_blocking=True),
            "attention_mask": attention_mask.to(self.device, non_blocking=True),
            "image_flags": torch.ones(batch_size, 1, device=self.device, dtype=torch.long),
        }
        for key, value in image_inputs.items():
            if torch.is_tensor(value):
                dtype = torch.bfloat16 if value.dtype.is_floating_point else value.dtype
                model_inputs[key] = value.to(self.device, dtype=dtype, non_blocking=True)
        return model_inputs

    def _resolve_chat_model(self) -> nn.Module:
        model = self.vla
        visited = set()
        while model is not None and id(model) not in visited:
            visited.add(id(model))
            if hasattr(model, "language_model") and hasattr(model, "extract_feature"):
                return model
            next_model = None
            for attr_name in ["model", "base_model", "module"]:
                candidate = getattr(model, attr_name, None)
                if candidate is not None and candidate is not model:
                    next_model = candidate
                    break
            model = next_model
        raise RuntimeError("Failed to resolve InternVLChatModel from wrapped VLA module")

    def _build_multimodal_prefix_embeds(
        self,
        model_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chat_model = self._resolve_chat_model()
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        image_flags = model_inputs["image_flags"].squeeze(-1)
        pixel_values = model_inputs["pixel_values"]

        input_embeds = chat_model.language_model.get_input_embeddings()(input_ids).clone()
        vit_embeds = chat_model.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]

        batch_size, seq_len, hidden_dim = input_embeds.shape
        flat_input_embeds = input_embeds.reshape(batch_size * seq_len, hidden_dim)
        flat_input_ids = input_ids.reshape(batch_size * seq_len)
        selected = flat_input_ids == chat_model.img_context_token_id
        vit_embeds = vit_embeds.reshape(-1, hidden_dim)
        token_count = min(int(selected.sum().item()), vit_embeds.shape[0])
        if token_count > 0:
            selected_indices = torch.nonzero(selected, as_tuple=False).squeeze(-1)[:token_count]
            flat_input_embeds[selected_indices] = vit_embeds[:token_count].to(flat_input_embeds.dtype)
        prefix_embeds = flat_input_embeds.reshape(batch_size, seq_len, hidden_dim)
        return prefix_embeds, attention_mask

    def env_actions_to_bin_indices(self, env_actions: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(env_actions, torch.Tensor):
            env_actions = env_actions.detach().cpu().numpy()
        env_actions = np.asarray(env_actions, dtype=np.float32)
        if env_actions.ndim == 1:
            env_actions = env_actions[None, :]
        env_actions = np.clip(env_actions, -1.0, 1.0)
        scaled = (env_actions + 1.0) * 0.5 * (self.num_action_bins - 1)
        return torch.from_numpy(np.rint(scaled).astype(np.int64))

    def bin_indices_to_env_actions(self, bin_indices: torch.Tensor) -> torch.Tensor:
        if bin_indices.ndim == 1:
            bin_indices = bin_indices.unsqueeze(0)
        bin_indices = bin_indices.to(self.device, dtype=torch.long)
        bin_indices = torch.clamp(bin_indices, 0, self.action_bin_centers.numel() - 1)
        env_actions = self.action_bin_centers.to(self.device)[bin_indices].to(torch.float32)
        return torch.nan_to_num(env_actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)

    def _extract_hidden_states(self, output: Any) -> torch.Tensor:
        if hasattr(output, "hidden_states") and output.hidden_states:
            return output.hidden_states[-1]
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state
        raise RuntimeError("InternVL forward pass did not return hidden states")

    def _run_language_model(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self._forward_language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return self._extract_hidden_states(output).to(torch.float32)

    def _forward_language_model(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        use_cache: bool,
        past_key_values: Any = None,
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Any:
        chat_model = self._resolve_chat_model()
        lm_embed_dtype = chat_model.language_model.get_input_embeddings().weight.dtype
        inputs_embeds = inputs_embeds.to(dtype=lm_embed_dtype)

        def forward_once():
            kwargs = dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            kwargs["use_cache"] = use_cache
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            if cache_position is not None:
                kwargs["cache_position"] = cache_position
            return chat_model.language_model(**kwargs)

        if self._vla_trainable:
            output = forward_once()
        else:
            with torch.no_grad():
                output = forward_once()
        return output

    def _build_state_tokens(self, agent_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_agents, _ = agent_states.shape
        flat_states = agent_states.reshape(batch_size * num_agents, -1)
        projected = self.state_projector(flat_states).reshape(batch_size, num_agents, self.hidden_dim)
        role_ids = torch.arange(num_agents, device=self.device, dtype=torch.long)
        role_embeds = self.state_role_embeddings(role_ids).unsqueeze(0)
        return projected + role_embeds

    def _build_joint_suffix_inputs(self, action_bins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_agents, action_dim = action_bins.shape
        suffix_tokens = []
        target_tokens = []
        for agent_idx in range(num_agents):
            marker = self.agent_marker_embeddings.weight[agent_idx].view(1, 1, -1).expand(batch_size, 1, -1)
            prev_tokens = action_bins[:, agent_idx, :-1]
            prev_embeds = self.action_token_embedding(prev_tokens) if action_dim > 1 else marker.new_zeros((batch_size, 0, marker.shape[-1]))
            suffix_tokens.append(torch.cat([marker, prev_embeds], dim=1))
            target_tokens.append(action_bins[:, agent_idx, :])
        return torch.cat(suffix_tokens, dim=1), torch.cat(target_tokens, dim=1)

    def _decode_joint_action_tokens(
        self,
        rgbs: Union[np.ndarray, torch.Tensor],
        agent_states: torch.Tensor,
        action_bins: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        model_inputs = self._prepare_policy_inputs(rgbs)
        prefix_embeds, prefix_attention_mask = self._build_multimodal_prefix_embeds(model_inputs)

        state_tensor = torch.as_tensor(agent_states, device=self.device, dtype=torch.float32)
        state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
        state_tokens = self._build_state_tokens(state_tensor).to(dtype=prefix_embeds.dtype)
        suffix_embeds, target_tokens = self._build_joint_suffix_inputs(action_bins.to(self.device, dtype=torch.long))
        suffix_embeds = suffix_embeds.to(dtype=prefix_embeds.dtype)

        full_embeds = torch.cat([prefix_embeds, state_tokens, suffix_embeds], dim=1)
        suffix_mask = torch.ones(
            (full_embeds.shape[0], state_tokens.shape[1] + suffix_embeds.shape[1]),
            device=self.device,
            dtype=prefix_attention_mask.dtype,
        )
        full_attention_mask = torch.cat([prefix_attention_mask, suffix_mask], dim=1)
        hidden_states = self._run_language_model(full_embeds, full_attention_mask)
        action_hidden = hidden_states[:, -suffix_embeds.shape[1]:, :]
        logits = self.action_head(action_hidden) * self.action_logit_scale
        return {
            "logits": logits,
            "targets": target_tokens,
            "state_features": state_tokens,
            "prefix_hidden": hidden_states[:, prefix_embeds.shape[1] - 1, :],
        }

    def _sample_joint_action_tokens_no_cache(
        self,
        rgbs: Union[np.ndarray, torch.Tensor],
        agent_states: torch.Tensor,
        *,
        deterministic: bool,
    ) -> Dict[str, torch.Tensor]:
        model_inputs = self._prepare_policy_inputs(rgbs)
        prefix_embeds, prefix_attention_mask = self._build_multimodal_prefix_embeds(model_inputs)

        state_tensor = torch.as_tensor(agent_states, device=self.device, dtype=torch.float32)
        state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
        state_tokens = self._build_state_tokens(state_tensor).to(dtype=prefix_embeds.dtype)
        batch_size = state_tokens.shape[0]

        suffix_chunks = []
        sampled_tokens = []
        step_log_probs = []
        step_entropies = []
        num_predictions = self.num_agents * self.env_action_dim

        for step in range(num_predictions):
            agent_idx = step // self.env_action_dim
            agent_pos = step % self.env_action_dim
            if agent_pos == 0:
                next_chunk = self.agent_marker_embeddings.weight[agent_idx].view(1, 1, -1).expand(batch_size, 1, -1)
            else:
                prev_token = sampled_tokens[-1]
                next_chunk = self.action_token_embedding(prev_token).unsqueeze(1)
            next_chunk = next_chunk.to(dtype=prefix_embeds.dtype)
            suffix_chunks.append(next_chunk)
            suffix_embeds = torch.cat(suffix_chunks, dim=1)
            full_embeds = torch.cat([prefix_embeds, state_tokens, suffix_embeds], dim=1)
            suffix_mask = torch.ones(
                (batch_size, state_tokens.shape[1] + suffix_embeds.shape[1]),
                device=self.device,
                dtype=prefix_attention_mask.dtype,
            )
            full_attention_mask = torch.cat([prefix_attention_mask, suffix_mask], dim=1)
            hidden_states = self._run_language_model(full_embeds, full_attention_mask)
            last_hidden = hidden_states[:, -1, :]
            logits = self.action_head(last_hidden) * self.action_logit_scale
            categorical = torch.distributions.Categorical(logits=logits)
            next_token = logits.argmax(dim=-1) if deterministic else categorical.sample()
            sampled_tokens.append(next_token)
            step_log_probs.append(categorical.log_prob(next_token))
            step_entropies.append(categorical.entropy())

        tokens = torch.stack(sampled_tokens, dim=1).reshape(batch_size, self.num_agents, self.env_action_dim)
        return {
            "tokens": tokens,
            "log_prob": torch.stack(step_log_probs, dim=1).reshape(batch_size, self.num_agents, self.env_action_dim).sum(dim=-1),
            "entropy": torch.stack(step_entropies, dim=1).reshape(batch_size, self.num_agents, self.env_action_dim).mean(dim=-1),
        }

    def _sample_joint_action_tokens(
        self,
        rgbs: Union[np.ndarray, torch.Tensor],
        agent_states: torch.Tensor,
        *,
        deterministic: bool,
    ) -> Dict[str, torch.Tensor]:
        model_inputs = self._prepare_policy_inputs(rgbs)
        prefix_embeds, prefix_attention_mask = self._build_multimodal_prefix_embeds(model_inputs)

        state_tensor = torch.as_tensor(agent_states, device=self.device, dtype=torch.float32)
        state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1e4, neginf=-1e4)
        state_tokens = self._build_state_tokens(state_tensor).to(dtype=prefix_embeds.dtype)
        batch_size = state_tokens.shape[0]

        prefix_state_embeds = torch.cat([prefix_embeds, state_tokens], dim=1)
        prefix_state_mask = torch.cat(
            [
                prefix_attention_mask,
                torch.ones(
                    (batch_size, state_tokens.shape[1]),
                    device=self.device,
                    dtype=prefix_attention_mask.dtype,
                ),
            ],
            dim=1,
        )
        prefix_position_ids = prefix_state_mask.cumsum(dim=1) - 1
        prefix_position_ids = prefix_position_ids.masked_fill(prefix_state_mask == 0, 0).to(dtype=torch.long)
        prefix_cache_position = torch.arange(prefix_state_embeds.shape[1], device=self.device, dtype=torch.long)
        prefix_output = self._forward_language_model(
            inputs_embeds=prefix_state_embeds,
            attention_mask=prefix_state_mask,
            use_cache=True,
            position_ids=prefix_position_ids,
            cache_position=prefix_cache_position,
        )
        past_key_values = getattr(prefix_output, "past_key_values", None)
        if past_key_values is None:
            return self._sample_joint_action_tokens_no_cache(
                rgbs=rgbs,
                agent_states=agent_states,
                deterministic=deterministic,
            )

        sampled_tokens = []
        step_log_probs = []
        step_entropies = []
        running_attention_mask = prefix_state_mask
        num_predictions = self.num_agents * self.env_action_dim

        for step in range(num_predictions):
            agent_idx = step // self.env_action_dim
            agent_pos = step % self.env_action_dim
            if agent_pos == 0:
                next_chunk = self.agent_marker_embeddings.weight[agent_idx].view(1, 1, -1).expand(batch_size, 1, -1)
            else:
                prev_token = sampled_tokens[-1]
                next_chunk = self.action_token_embedding(prev_token).unsqueeze(1)
            next_chunk = next_chunk.to(dtype=prefix_embeds.dtype)
            running_attention_mask = torch.cat(
                [
                    running_attention_mask,
                    torch.ones((batch_size, 1), device=self.device, dtype=prefix_attention_mask.dtype),
                ],
                dim=1,
            )
            position_ids = running_attention_mask.cumsum(dim=1) - 1
            position_ids = position_ids.masked_fill(running_attention_mask == 0, 0).to(dtype=torch.long)[:, -1:]
            cache_position = torch.full(
                (1,),
                running_attention_mask.shape[1] - 1,
                device=self.device,
                dtype=torch.long,
            )
            output = self._forward_language_model(
                inputs_embeds=next_chunk,
                attention_mask=running_attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
                position_ids=position_ids,
                cache_position=cache_position,
            )
            past_key_values = getattr(output, "past_key_values", None)
            if past_key_values is None:
                return self._sample_joint_action_tokens_no_cache(
                    rgbs=rgbs,
                    agent_states=agent_states,
                    deterministic=deterministic,
                )
            last_hidden = self._extract_hidden_states(output)[:, -1, :].to(torch.float32)
            logits = self.action_head(last_hidden) * self.action_logit_scale
            categorical = torch.distributions.Categorical(logits=logits)
            next_token = logits.argmax(dim=-1) if deterministic else categorical.sample()
            sampled_tokens.append(next_token)
            step_log_probs.append(categorical.log_prob(next_token))
            step_entropies.append(categorical.entropy())

        tokens = torch.stack(sampled_tokens, dim=1).reshape(batch_size, self.num_agents, self.env_action_dim)
        return {
            "tokens": tokens,
            "log_prob": torch.stack(step_log_probs, dim=1).reshape(batch_size, self.num_agents, self.env_action_dim).sum(dim=-1),
            "entropy": torch.stack(step_entropies, dim=1).reshape(batch_size, self.num_agents, self.env_action_dim).mean(dim=-1),
        }

    def get_action_and_stats(
        self,
        rgbs: Union[np.ndarray, torch.Tensor],
        states: torch.Tensor,
        action_bins: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if action_bins is None:
            sampled = self._sample_joint_action_tokens(rgbs, states, deterministic=deterministic)
            selected_bins = sampled["tokens"]
            log_prob = sampled["log_prob"]
            entropy = sampled["entropy"]
            teacher_forced = self._decode_joint_action_tokens(rgbs, states, selected_bins)
            logits = teacher_forced["logits"]
            state_features = teacher_forced["state_features"]
            prompt_hidden = teacher_forced["prefix_hidden"]
        else:
            selected_bins = action_bins.to(self.device, dtype=torch.long)
            teacher_forced = self._decode_joint_action_tokens(rgbs, states, selected_bins)
            logits = teacher_forced["logits"]
            targets = teacher_forced["targets"]
            categorical = torch.distributions.Categorical(logits=logits)
            log_prob = categorical.log_prob(targets).reshape(targets.shape[0], self.num_agents, self.env_action_dim).sum(dim=-1)
            entropy = categorical.entropy().reshape(targets.shape[0], self.num_agents, self.env_action_dim).mean(dim=-1)
            state_features = teacher_forced["state_features"]
            prompt_hidden = teacher_forced["prefix_hidden"]
        env_actions = self.bin_indices_to_env_actions(selected_bins)
        return {
            "env_actions": env_actions,
            "log_prob": log_prob,
            "entropy": entropy,
            "action_bins": selected_bins,
            "action_logits": logits,
            "prompt_hidden": prompt_hidden,
            "state_features": state_features,
        }

    def sample_env_actions(
        self,
        rgbs: Union[np.ndarray, torch.Tensor],
        states: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        sampled = self._sample_joint_action_tokens(rgbs, states, deterministic=deterministic)
        return self.bin_indices_to_env_actions(sampled["tokens"])


class MultiAgentVLAAdapterAgent(nn.Module):
    def __init__(
        self,
        agent_names: List[str],
        state_dim: int,
        global_state_dim: int,
        action_dim: int,
        model_dir: Union[str, Path],
        normalize_state: bool = True,
        freeze_vla_backbone: bool = False,
        critic_hidden_dim: int = 512,
        attention_implementation: str = "sdpa",
        image_size: Optional[int] = 112,
        use_vla_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.agent_names = list(agent_names)
        self.num_agents = len(self.agent_names)
        self.state_dim = state_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim

        self.actor = SharedInternVLActor(
            model_dir=Path(model_dir),
            state_dim=state_dim,
            env_action_dim=action_dim,
            attention_implementation=attention_implementation,
            image_size=image_size,
            use_lora=use_vla_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.actor.configure_trainable_modules(not freeze_vla_backbone)

        self.actor_feature_placeholders = nn.ModuleDict({name: nn.Identity() for name in self.agent_names})

        if normalize_state:
            self.actor_state_rms = nn.ModuleDict(
                {name: RunningMeanStd(shape=(state_dim,)) for name in self.agent_names}
            )
            self.critic_state_rms = RunningMeanStd(shape=(global_state_dim,))
        else:
            self.actor_state_rms = None
            self.critic_state_rms = None

        hidden_dim = self.actor.hidden_dim
        self.critic_state_encoder = nn.Sequential(
            nn.LayerNorm(global_state_dim),
            nn.Linear(global_state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(dtype=torch.float32)
        critic_input_dim = hidden_dim * (1 + 3 * self.num_agents)
        self.critic = nn.Sequential(
            nn.LayerNorm(critic_input_dim),
            nn.Linear(critic_input_dim, critic_hidden_dim),
            nn.GELU(),
            nn.Linear(critic_hidden_dim, 1),
        ).to(dtype=torch.float32)

    def configure_trainable_modules(self, freeze_vla_backbone: bool) -> None:
        self.actor.configure_trainable_modules(not freeze_vla_backbone)
        for module in [self.critic_state_encoder, self.critic]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def trainable_parameter_summary(self) -> Dict[str, Tuple[int, int]]:
        modules = {
            "vla": self.actor.vla,
            "state_projector": self.actor.state_projector,
            "state_role_embeddings": self.actor.state_role_embeddings,
            "agent_marker_embeddings": self.actor.agent_marker_embeddings,
            "action_token_embedding": self.actor.action_token_embedding,
            "action_head": self.actor.action_head,
            "critic_state_encoder": self.critic_state_encoder,
            "critic": self.critic,
        }
        summary = {}
        for name, module in modules.items():
            total = sum(parameter.numel() for parameter in module.parameters())
            trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
            summary[name] = (total, trainable)
        return summary

    def _normalize_actor_state(self, state: torch.Tensor, agent_name: str) -> torch.Tensor:
        if self.actor_state_rms is None:
            return state
        return self.actor_state_rms[agent_name](state)

    def _normalize_global_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.critic_state_rms is None:
            return state
        return self.critic_state_rms(state)

    def _run_shared_actor(
        self,
        batch: Dict[str, Any],
        actions_input: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
        action_bins_input: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
        deterministic: bool = False,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        List[torch.Tensor],
    ]:
        rgb = batch["rgb"]

        normalized_states = []
        for name in self.agent_names:
            state = batch[f"agent_states_{name}"]
            state = state.to(device=self.actor.device, dtype=torch.float32)
            normalized_states.append(self._normalize_actor_state(state, name))
        joint_states = torch.stack(normalized_states, dim=1)

        action_bins = None
        if action_bins_input is not None:
            action_bins = torch.stack(
                [
                    torch.as_tensor(action_bins_input[name], device=self.actor.device, dtype=torch.long)
                    for name in self.agent_names
                ],
                dim=1,
            )
        elif actions_input is not None:
            action_bins = torch.stack(
                [self.actor.env_actions_to_bin_indices(actions_input[name]) for name in self.agent_names],
                dim=1,
            ).to(self.actor.device)

        actor_out = self.actor.get_action_and_stats(
            rgbs=rgb,
            states=joint_states,
            action_bins=action_bins,
            deterministic=deterministic,
        )

        batch_size = joint_states.shape[0]
        actions_out = {}
        log_probs = {}
        entropies = {}
        action_bins_out = {}
        critic_agent_features = []
        for idx, name in enumerate(self.agent_names):
            actions_out[name] = actor_out["env_actions"][:, idx, :].detach().cpu().numpy()
            log_probs[name] = actor_out["log_prob"][:, idx]
            entropies[name] = actor_out["entropy"][:, idx]
            action_bins_out[name] = actor_out["action_bins"][:, idx, :]
            prompt_hidden = actor_out["prompt_hidden"]
            state_feature = actor_out["state_features"][:, idx, :]
            critic_agent_features.append(torch.cat([prompt_hidden, state_feature, state_feature], dim=-1))
        return actions_out, log_probs, entropies, action_bins_out, critic_agent_features

    def _compute_value(self, batch: Dict[str, Any], critic_agent_features: List[torch.Tensor]) -> torch.Tensor:
        global_state = batch["global_state"].to(device=self.actor.device, dtype=torch.float32)
        global_state = self._normalize_global_state(global_state)
        global_feature = self.critic_state_encoder(global_state)
        critic_input = torch.cat([global_feature] + critic_agent_features, dim=-1)
        return self.critic(critic_input).squeeze(-1)

    def get_action_and_value(
        self,
        batch: Dict[str, Any],
        actions_input=None,
        action_bins_input=None,
        return_action_bins: bool = False,
    ):
        actions_out, log_probs, entropies, action_bins_out, critic_agent_features = self._run_shared_actor(
            batch,
            actions_input=actions_input,
            action_bins_input=action_bins_input,
            deterministic=False,
        )
        value = self._compute_value(batch, critic_agent_features)
        if return_action_bins:
            return actions_out, log_probs, entropies, value, action_bins_out
        return actions_out, log_probs, entropies, value

    @torch.no_grad()
    def get_action(self, batch: Dict[str, Any], deterministic: bool = False) -> Dict[str, np.ndarray]:
        rgb = batch["rgb"]
        normalized_states = []
        for name in self.agent_names:
            state = batch[f"agent_states_{name}"]
            state = state.to(device=self.actor.device, dtype=torch.float32)
            normalized_states.append(self._normalize_actor_state(state, name))
        joint_states = torch.stack(normalized_states, dim=1)
        env_actions = self.actor.sample_env_actions(
            rgbs=rgb,
            states=joint_states,
            deterministic=deterministic,
        )
        actions_out = {}
        for idx, name in enumerate(self.agent_names):
            actions_out[name] = env_actions[:, idx, :].detach().cpu().numpy()
        return actions_out

    def get_value(self, batch: Dict[str, Any]) -> torch.Tensor:
        _, _, _, _, critic_agent_features = self._run_shared_actor(
            batch,
            actions_input=None,
            deterministic=True,
        )
        return self._compute_value(batch, critic_agent_features)

    def forward(self, batch: Dict[str, Any]):
        _, log_probs, _, values = self.get_action_and_value(batch)
        return log_probs, values

    @torch.no_grad()
    def update_state_stats(self, obs: Dict[str, Any]) -> None:
        parsed = build_batch_from_obs(obs, self.agent_names)
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].update(
                    parsed[f"agent_states_{name}"].to(device=self.actor.device, dtype=torch.float32)
                )
        if self.critic_state_rms is not None:
            self.critic_state_rms.update(parsed["global_state"].to(device=self.actor.device, dtype=torch.float32))

    def freeze_state_stats(self) -> None:
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].freeze()
        if self.critic_state_rms is not None:
            self.critic_state_rms.freeze()

    def unfreeze_state_stats(self) -> None:
        if self.actor_state_rms is not None:
            for name in self.agent_names:
                self.actor_state_rms[name].unfreeze()
        if self.critic_state_rms is not None:
            self.critic_state_rms.unfreeze()

    def checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        state = self.state_dict()
        if not self.actor.use_lora:
            return state
        filtered_state = {}
        for key, value in state.items():
            if key.startswith("actor.vla."):
                if "lora_" not in key.lower():
                    continue
            filtered_state[key] = value
        return filtered_state

    def load_checkpoint_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if self.actor.use_lora:
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            unexpected = [key for key in unexpected if "lora_" not in key.lower()]
            missing = [
                key for key in missing
                if not (key.startswith("actor.vla.") and "lora_" not in key.lower())
            ]
            if unexpected:
                raise RuntimeError(f"Unexpected keys when loading LoRA checkpoint: {unexpected}")
            if missing:
                raise RuntimeError(f"Missing keys when loading LoRA checkpoint: {missing}")
            return
        self.load_state_dict(state_dict)


# Backward-compatible name used by existing MAPPO training scripts.
MultiAgentVLAAdapterMAPPOAgent = MultiAgentVLAAdapterAgent


def build_optimizer(args, agent: MultiAgentVLAAdapterMAPPOAgent) -> torch.optim.Optimizer:
    param_groups = [
        {
            "params": list(agent.actor.vla.parameters()),
            "lr": 0.0 if getattr(args, "freeze_vla_backbone", False) else args.backbone_learning_rate,
            "group_name": "vla",
        },
        {
            "params": list(agent.actor.state_projector.parameters()),
            "lr": args.state_learning_rate,
            "group_name": "state_projector",
        },
        {
            "params": list(agent.actor.state_role_embeddings.parameters())
            + list(agent.actor.agent_marker_embeddings.parameters())
            + list(agent.actor.action_token_embedding.parameters())
            + list(agent.actor.action_head.parameters()),
            "lr": args.head_learning_rate,
            "group_name": "action_tokens",
        },
        {
            "params": list(agent.critic_state_encoder.parameters()),
            "lr": args.value_head_learning_rate,
            "group_name": "critic_state_encoder",
        },
        {
            "params": list(agent.critic.parameters()),
            "lr": args.value_head_learning_rate,
            "group_name": "critic",
        },
    ]
    return torch.optim.AdamW(param_groups, eps=1e-5, weight_decay=args.weight_decay)
