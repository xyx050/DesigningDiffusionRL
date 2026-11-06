import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml


DEFAULT_HPSV3_REPO = Path("third_party/HPSv3")
DEFAULT_HPSV3_CHECKPOINT = Path("checkpoints/HPSv3/HPSv3.safetensors")
DEFAULT_BASE_MODEL_CANDIDATES = (
    Path("checkpoints/Qwen2-VL-7B-Instruct"),
    Path("checkpoints/Qwen/Qwen2-VL-7B-Instruct"),
    Path("checkpoints/Qwen2.5-VL-7B-Instruct"),
    Path("checkpoints/Qwen/Qwen2.5-VL-7B-Instruct"),
)


def _as_path(value):
    return Path(value).expanduser()


def _resolve_hpsv3_repo():
    return _as_path(os.environ.get("HPSV3_REPO", DEFAULT_HPSV3_REPO))


def _resolve_checkpoint_path():
    return _as_path(os.environ.get("HPSV3_CHECKPOINT", DEFAULT_HPSV3_CHECKPOINT))


def _resolve_config_path(repo_path):
    return _as_path(
        os.environ.get("HPSV3_CONFIG", repo_path / "hpsv3" / "config" / "HPSv3_7B.yaml")
    )


def _find_local_base_model():
    env_base_model = os.environ.get("HPSV3_BASE_MODEL")
    if env_base_model:
        return env_base_model

    for candidate in DEFAULT_BASE_MODEL_CANDIDATES:
        if candidate.is_dir():
            return str(candidate)
    return None


def _is_existing_local_path(path_or_repo_id):
    if not path_or_repo_id:
        return False
    return Path(path_or_repo_id).expanduser().exists()


def _validate_local_base_model(base_model):
    base_path = Path(base_model).expanduser()
    missing = []
    if not (base_path / "config.json").is_file():
        missing.append("config.json")
    if not any((base_path / name).is_file() for name in ("preprocessor_config.json", "processor_config.json")):
        missing.append("preprocessor_config.json or processor_config.json")
    if not any((base_path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        missing.append("tokenizer.json, tokenizer.model, or vocab.json")

    if missing:
        raise FileNotFoundError(
            f"HPSv3 base model directory is incomplete: {base_path}. Missing: {missing}."
        )


def _allow_hf_download():
    return os.environ.get("HPSV3_ALLOW_HF_DOWNLOAD", "").lower() in {"1", "true", "yes"}


def _resolve_device(device):
    env_device = os.environ.get("HPSV3_DEVICE")
    if env_device:
        return torch.device(env_device)
    if os.environ.get("HPSV3_USE_TRAINING_DEVICE", "").lower() in {"1", "true", "yes"}:
        return torch.device(device)

    training_device = torch.device(device)
    if training_device.type != "cuda" or not torch.cuda.is_available():
        return training_device

    default_local_rank = training_device.index if training_device.index is not None else torch.cuda.current_device()
    local_rank = int(os.environ.get("LOCAL_RANK", default_local_rank))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    device_count = torch.cuda.device_count()
    reward_device_index = local_rank + local_world_size
    if reward_device_index < device_count:
        return torch.device("cuda", reward_device_index)

    return training_device


def _ensure_hpsv3_transformers_compatibility():
    from transformers import image_utils

    if not hasattr(image_utils, "VideoInput"):
        from transformers.video_utils import VideoInput

        image_utils.VideoInput = VideoInput


def _patch_hpsv3_state_dict_prefix_compatibility():
    from hpsv3.model.qwen2vl_trainer import Qwen2VLRewardModelBT

    if getattr(Qwen2VLRewardModelBT.load_state_dict, "_flow_grpo_hpsv3_prefix_patch", False):
        return

    original_load_state_dict = Qwen2VLRewardModelBT.load_state_dict

    def load_state_dict_with_prefix_compatibility(self, state_dict, *args, **kwargs):
        expected_keys = self.state_dict().keys()
        uses_new_language_prefix = any(key.startswith("model.language_model.") for key in expected_keys)
        uses_new_visual_prefix = any(key.startswith("model.visual.") for key in expected_keys)

        if (
            isinstance(state_dict, dict)
            and (
                (
                    uses_new_visual_prefix
                    and any(key.startswith("visual.") for key in state_dict)
                    and not any(key.startswith("model.visual.") for key in state_dict)
                )
                or (
                    uses_new_language_prefix
                    and any(key.startswith("model.layers.") or key == "model.embed_tokens.weight" for key in state_dict)
                    and not any(key.startswith("model.language_model.") for key in state_dict)
                )
            )
        ):
            remapped_state_dict = {}
            for key, value in state_dict.items():
                if uses_new_visual_prefix and key.startswith("visual."):
                    key = f"model.{key}"
                elif (
                    uses_new_language_prefix
                    and key.startswith("model.")
                    and not key.startswith(("model.visual.", "model.language_model."))
                ):
                    key = f"model.language_model.{key[len('model.'):]}"
                remapped_state_dict[key] = value
            state_dict = remapped_state_dict

        return original_load_state_dict(self, state_dict, *args, **kwargs)

    load_state_dict_with_prefix_compatibility._flow_grpo_hpsv3_prefix_patch = True
    Qwen2VLRewardModelBT.load_state_dict = load_state_dict_with_prefix_compatibility


def _ensure_hpsv3_model_structure_compatibility(model):
    inner_model = getattr(model, "model", None)
    language_model = getattr(inner_model, "language_model", None)
    if (
        inner_model is not None
        and language_model is not None
        and not hasattr(inner_model, "embed_tokens")
        and hasattr(language_model, "embed_tokens")
    ):
        inner_model.embed_tokens = language_model.embed_tokens


def _has_configured_accelerate_state():
    try:
        from accelerate.state import AcceleratorState, PartialState
    except ImportError:
        return False

    return bool(AcceleratorState._shared_state or PartialState._shared_state)


def _resolve_base_model(config_path):
    base_model = _find_local_base_model()
    if base_model:
        if _is_existing_local_path(base_model):
            _validate_local_base_model(base_model)
        elif not _allow_hf_download():
            raise RuntimeError(
                f"HPSV3_BASE_MODEL points to a Hugging Face repo id, not a local path: {base_model}. "
                "Pass a local Qwen-VL directory, or set HPSV3_ALLOW_HF_DOWNLOAD=1."
            )
        return base_model

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    configured_base_model = config.get("model_name_or_path")
    if configured_base_model and _is_existing_local_path(configured_base_model):
        _validate_local_base_model(configured_base_model)
        return None

    if _allow_hf_download():
        return None

    raise RuntimeError(
        "No local Qwen-VL base model was found for HPSv3. Set HPSV3_BASE_MODEL to a local "
        "Qwen/Qwen2-VL-7B-Instruct directory, or set HPSV3_ALLOW_HF_DOWNLOAD=1 to allow "
        "transformers to download it."
    )


def _config_with_base_model(config_path, base_model, device):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if base_model:
        config["model_name_or_path"] = base_model
    if torch.device(device).type == "cpu":
        config["disable_flash_attn2"] = True
    if _has_configured_accelerate_state():
        config["accelerator_config"] = {"use_configured_state": True}
    else:
        config.pop("accelerator_config", None)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="flow_grpo_hpsv3_",
        delete=False,
        encoding="utf-8",
    )
    with tmp:
        yaml.safe_dump(config, tmp, sort_keys=False)

    return tmp.name, tmp.name


class HPSv3Scorer(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.training_device = torch.device(device)
        self.device = _resolve_device(device)
        self.uses_training_device = self.device == self.training_device
        self.inferencer = None
        self.temporary_config_path = None

        repo_path = _resolve_hpsv3_repo()
        checkpoint_path = _resolve_checkpoint_path()
        config_path = _resolve_config_path(repo_path)

        if not repo_path.is_dir():
            raise FileNotFoundError(f"HPSv3 repo not found: {repo_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"HPSv3 checkpoint not found: {checkpoint_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"HPSv3 config not found: {config_path}")

        repo_path_str = str(repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)

        _ensure_hpsv3_transformers_compatibility()
        base_model = _resolve_base_model(config_path)
        resolved_config_path, temporary_config_path = _config_with_base_model(config_path, base_model, self.device)

        self.checkpoint_path = checkpoint_path
        self.resolved_config_path = resolved_config_path
        self.temporary_config_path = temporary_config_path

    def _ensure_inferencer(self):
        if self.inferencer is not None:
            self.inferencer.device = self.device
            self.inferencer.model.to(self.device)
            return

        from hpsv3 import HPSv3RewardInferencer

        _patch_hpsv3_state_dict_prefix_compatibility()
        self.inferencer = HPSv3RewardInferencer(
            config_path=self.resolved_config_path,
            checkpoint_path=str(self.checkpoint_path),
            device=self.device,
        )
        _ensure_hpsv3_model_structure_compatibility(self.inferencer.model)

    def offload(self):
        if self.inferencer is None:
            return

        self.inferencer.model.to("cpu")
        self.inferencer.device = torch.device("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def __del__(self):
        temporary_config_path = getattr(self, "temporary_config_path", None)
        if temporary_config_path and os.path.exists(temporary_config_path):
            os.unlink(temporary_config_path)

    @torch.no_grad()
    def __call__(self, prompts, images):
        self._ensure_inferencer()
        prompts = list(prompts)
        rewards = self.inferencer.reward(prompts, image_paths=images)

        if isinstance(rewards, torch.Tensor):
            if rewards.ndim == 2:
                rewards = rewards[:, 0]
            return rewards.detach().float().cpu().tolist()

        scores = []
        for reward in rewards:
            if isinstance(reward, torch.Tensor):
                reward = reward[0] if reward.ndim > 0 else reward
                scores.append(float(reward.detach().float().cpu().item()))
            elif isinstance(reward, (list, tuple)):
                scores.append(float(reward[0]))
            else:
                scores.append(float(reward))
        return scores
