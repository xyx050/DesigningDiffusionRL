"""NFT (Negative Flow Training) for Qwen-Image.

The Qwen-Image model, prompt encoder, packed latent representation, and FSDP
execution follow train_qwenimage.py.  The rollout policy, objective, and
old-policy update follow train_sd3_nft.py.
"""

from collections import defaultdict
from concurrent import futures
from functools import partial
import contextlib
import datetime
import hashlib
import json
import logging
import os
import random
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

from absl import app, flags
from diffusers import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from ml_collections import config_flags
import numpy as np
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
)
from PIL import Image
from safetensors.torch import load_file, save_file
import torch
import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import tqdm
from flow_grpo.logging_utils import ExperimentLogger

wandb = ExperimentLogger()

import flow_grpo.rewards
from flow_grpo.diffusers_patch.nft_solver import run_sampling
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.fsdp_utils import (
    FSDPConfig,
    fsdp_wrapper,
    init_distributed,
    register_optimizer_offload_hooks,
)
from flow_grpo.resume_utils import (
    COMPLETE_NAME,
    META_NAME,
    expected_metadata,
    find_latest_checkpoint,
    load_metadata,
    load_rng_state_dict,
    rng_state_dict,
    torch_load,
)
from flow_grpo.stat_tracking import PerPromptStatTracker


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

EVAL_PROMPT_COUNT = 2048
EVAL_LOG_IMAGE_COUNT = 30
QWEN_NFT_TRAINER_NAME = "train_qwenimage_nft"
RESUME_ADAPTERS_NAME = "resume_adapters.safetensors"
RESUME_OPTIMIZER_NAME = "resume_optimizer.pt"
RESUME_EMA_NAME = "resume_ema.pt"
RESUME_TRAINING_STATE_NAME = "resume_training_state.pt"
CHECKPOINT_COPY_BUFFER_SIZE = 16 * 1024 * 1024


def checkpoint_local_tmp_dir():
    path = os.environ.get("CHECKPOINT_LOCAL_TMPDIR")
    if not path:
        cache_dir = os.environ.get("CACHE_DIR")
        path = (
            os.path.join(cache_dir, "checkpoint_tmp")
            if cache_dir
            else tempfile.gettempdir()
        )
    os.makedirs(path, exist_ok=True)
    return path


def copy_file_streaming(source_path, destination_path):
    with open(source_path, "rb") as source:
        with open(destination_path, "wb") as destination:
            shutil.copyfileobj(
                source,
                destination,
                length=CHECKPOINT_COPY_BUFFER_SIZE,
            )


def save_safetensors_staged(state_dict, destination_path, metadata=None):
    file_descriptor, local_path = tempfile.mkstemp(
        prefix="qwenimage_nft_checkpoint_",
        suffix=".safetensors",
        dir=checkpoint_local_tmp_dir(),
    )
    os.close(file_descriptor)
    try:
        save_file(state_dict, local_path, metadata=metadata)
        copy_file_streaming(local_path, destination_path)
        if os.path.getsize(local_path) != os.path.getsize(destination_path):
            raise IOError(
                f"Incomplete checkpoint copy to {destination_path}: "
                f"local={os.path.getsize(local_path)}, "
                f"destination={os.path.getsize(destination_path)}"
            )
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


@contextlib.contextmanager
def load_safetensors_staged(source_path):
    file_descriptor, local_path = tempfile.mkstemp(
        prefix="qwenimage_nft_resume_",
        suffix=".safetensors",
        dir=checkpoint_local_tmp_dir(),
    )
    try:
        with os.fdopen(file_descriptor, "wb") as destination:
            with open(source_path, "rb") as source:
                shutil.copyfileobj(
                    source,
                    destination,
                    length=CHECKPOINT_COPY_BUFFER_SIZE,
                )
        yield load_file(local_path, device="cpu")
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


class ImmediateResult:
    """A Future-compatible wrapper for rewards evaluated synchronously."""

    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


def _apply_fix_config(config):
    if "fix" not in config or not config.fix:
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gather_tensor(tensor, world_size):
    if world_size == 1:
        return tensor

    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return torch.cat(gather_list)


def gather_prompt_strings(prompts, world_size):
    if world_size == 1:
        return list(prompts)

    gathered_prompts = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_prompts, list(prompts))
    return [
        prompt
        for prompts_from_rank in gathered_prompts
        for prompt in prompts_from_rank
    ]


def set_seed(seed, device_specific=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_specific and torch.cuda.is_available():
        rank = dist.get_rank() if dist.is_initialized() else 0
        torch.cuda.manual_seed_all(seed + rank)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r", encoding="utf-8") as file:
            self.prompts = [line.strip() for line in file.readlines()]
        if (
            split == "test"
            and os.path.basename(os.path.normpath(dataset)) == "pickscore"
        ):
            if len(self.prompts) < EVAL_PROMPT_COUNT:
                raise ValueError(
                    f"Expected at least {EVAL_PROMPT_COUNT} eval prompts in "
                    f"{self.file_path}, found {len(self.prompts)}."
                )
            self.prompts = self.prompts[:EVAL_PROMPT_COUNT]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as file:
            self.metadatas = [json.loads(line) for line in file]
        self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, (
            f"k can not divide n*b, k{self.k}-num_replicas"
            f"{self.num_replicas}-batch_size{self.batch_size}"
        )
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(
                len(self.dataset), generator=generator
            )[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(
                len(repeated_indices), generator=generator
            ).tolist()
            shuffled_samples = [repeated_indices[idx] for idx in shuffled_indices]

            start = self.rank * self.batch_size
            end = start + self.batch_size
            yield shuffled_samples[start:end]

    def set_epoch(self, epoch):
        self.epoch = epoch


def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], "big")
        seed = (base_seed + prompt_hash_int) % (2**31)
        generators.append(torch.Generator().manual_seed(seed))
    return generators


def create_train_permutation(
    num_samples,
    seed,
    epoch,
    inner_epoch,
    num_inner_epochs,
    device,
):
    generator = torch.Generator()
    generator.manual_seed(
        seed + epoch * num_inner_epochs + inner_epoch
    )
    return torch.randperm(num_samples, generator=generator).to(device)


def return_decay(step, decay_type):
    """Match the old-adapter decay schedule in train_sd3_nft.py."""
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")

    if step < flat:
        return 0.0
    return min((step - flat) * uprate, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    _, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_ratio = np.count_nonzero(prompt_std_devs == 0) / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def get_transformer_layer_cls():
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenImageTransformerBlock,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLDecoderLayer,
        Qwen2_5_VLVisionBlock,
    )

    return {
        QwenImageTransformerBlock,
        Qwen2_5_VLVisionBlock,
        Qwen2_5_VLDecoderLayer,
    }


def collect_trainable_adapter_parameters(transformer, adapter_name):
    transformer.set_adapter(adapter_name)
    return [parameter for parameter in transformer.parameters() if parameter.requires_grad]


def collect_named_adapter_parameters(transformer, parameters):
    parameter_ids = {id(parameter) for parameter in parameters}
    named_parameters = [
        (name, parameter)
        for name, parameter in transformer.named_parameters()
        if id(parameter) in parameter_ids
    ]
    if len(named_parameters) != len(parameters):
        raise RuntimeError(
            "Could not recover stable names for every default adapter parameter."
        )
    return named_parameters


def collect_lora_leaf_modules(transformer):
    """Keep LoRA parameters replicated so adapter blends remain well-defined."""
    modules = []
    for module_name, module in transformer.named_modules():
        is_lora_projection = ".lora_A." in module_name or ".lora_B." in module_name
        if is_lora_projection and any(
            True for _ in module.parameters(recurse=False)
        ):
            modules.append(module)
    if not modules:
        raise RuntimeError("No LoRA leaf modules were found for FSDP exclusion.")
    return modules


def validate_parameter_pairs(source_parameters, target_parameters, target_name):
    if len(source_parameters) != len(target_parameters):
        raise RuntimeError(
            f"default/{target_name} adapter parameter count mismatch: "
            f"{len(source_parameters)} != {len(target_parameters)}"
        )
    for index, (source, target) in enumerate(
        zip(source_parameters, target_parameters, strict=True)
    ):
        if source.shape != target.shape:
            raise RuntimeError(
                f"default/{target_name} adapter shape mismatch at {index}: "
                f"{tuple(source.shape)} != {tuple(target.shape)}"
            )
        if source is target:
            raise RuntimeError(
                f"default/{target_name} adapter unexpectedly shares parameter {index}"
            )


@torch.no_grad()
def copy_adapter_parameters(source_parameters, target_parameters):
    for source, target in zip(
        source_parameters, target_parameters, strict=True
    ):
        target.data.copy_(source.detach().data)


@torch.no_grad()
def blend_adapter_parameters(source_parameters, target_parameters, decay):
    for source, target in zip(
        source_parameters, target_parameters, strict=True
    ):
        target.data.mul_(decay).add_(source.detach().data, alpha=1.0 - decay)


def synchronize_replicated_gradients(parameters, world_size):
    if world_size == 1:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


@torch.no_grad()
def qwenimage_nft_pipeline(
    pipeline,
    prompt: Union[str, List[str]],
    negative_prompt: Optional[Union[str, List[str]]] = None,
    true_cfg_scale: float = 1.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    output_type: str = "pil",
    attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    noise_level: float = 0.7,
    deterministic: bool = True,
    solver: str = "dpm2",
):
    """Run Qwen-Image with the deterministic solvers used by SD3 NFT."""
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor
    if isinstance(prompt, str):
        prompt = [prompt]
    batch_size = len(prompt)
    if negative_prompt is None:
        negative_prompt = [" "] * batch_size
    elif isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt] * batch_size
    if len(negative_prompt) != batch_size:
        raise ValueError("negative_prompt must have the same batch size as prompt")

    pipeline.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=max_sequence_length,
    )
    pipeline._guidance_scale = None
    pipeline._attention_kwargs = attention_kwargs or {}
    pipeline._current_timestep = None
    pipeline._interrupt = False

    device = pipeline._execution_device
    do_true_cfg = true_cfg_scale > 1.0
    prompts_to_encode = (
        prompt + negative_prompt if do_true_cfg else prompt
    )
    prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
        prompt=prompts_to_encode,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
    if prompt_embeds_mask is None:
        prompt_embeds_mask = torch.ones(
            prompt_embeds.shape[:2],
            device=prompt_embeds.device,
            dtype=torch.long,
        )
    if do_true_cfg:
        prompt_embeds, negative_prompt_embeds = prompt_embeds.chunk(2, dim=0)
        prompt_embeds_mask, negative_prompt_embeds_mask = (
            prompt_embeds_mask.chunk(2, dim=0)
        )
    else:
        negative_prompt_embeds = prompt_embeds.new_empty(
            (batch_size, 0, prompt_embeds.shape[-1])
        )
        negative_prompt_embeds_mask = prompt_embeds_mask.new_empty(
            (batch_size, 0)
        )

    num_channels_latents = pipeline.transformer.config.in_channels // 4
    latents = pipeline.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    img_shapes = [
        [
            (
                1,
                height // pipeline.vae_scale_factor // 2,
                width // pipeline.vae_scale_factor // 2,
            )
        ]
    ] * batch_size

    if sigmas is None:
        sigmas = np.linspace(
            1.0, 1.0 / num_inference_steps, num_inference_steps
        )
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    pipeline._num_timesteps = len(timesteps)
    sigma_schedule = pipeline.scheduler.sigmas.float()

    def velocity_prediction(current_latents, sigma):
        if do_true_cfg:
            latent_model_input = torch.cat(
                [current_latents, current_latents], dim=0
            )
            encoder_hidden_states = torch.cat(
                [prompt_embeds, negative_prompt_embeds], dim=0
            )
            encoder_hidden_states_mask = torch.cat(
                [prompt_embeds_mask, negative_prompt_embeds_mask], dim=0
            )
            current_img_shapes = img_shapes * 2
        else:
            latent_model_input = current_latents
            encoder_hidden_states = prompt_embeds
            encoder_hidden_states_mask = prompt_embeds_mask
            current_img_shapes = img_shapes

        timestep = sigma.expand(latent_model_input.shape[0]).to(
            device=current_latents.device,
            dtype=current_latents.dtype,
        )
        if pipeline.transformer.config.guidance_embeds:
            guidance = torch.full(
                [latent_model_input.shape[0]],
                true_cfg_scale,
                device=current_latents.device,
                dtype=torch.float32,
            )
        else:
            guidance = None

        prediction = pipeline.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            encoder_hidden_states=encoder_hidden_states,
            img_shapes=current_img_shapes,
            attention_kwargs=pipeline.attention_kwargs,
            return_dict=False,
        )[0]
        if do_true_cfg:
            conditional_prediction, negative_prediction = prediction.chunk(
                2, dim=0
            )
            combined_prediction = negative_prediction + true_cfg_scale * (
                conditional_prediction - negative_prediction
            )
            conditional_norm = torch.norm(
                conditional_prediction, dim=-1, keepdim=True
            )
            combined_norm = torch.norm(
                combined_prediction, dim=-1, keepdim=True
            )
            prediction = combined_prediction * (
                conditional_norm / combined_norm
            )
        return prediction

    final_latents, all_latents, _ = run_sampling(
        velocity_prediction,
        latents,
        sigma_schedule,
        solver=solver,
        determistic=deterministic,
        eta=noise_level,
    )

    unpacked_latents = pipeline._unpack_latents(
        final_latents, height, width, pipeline.vae_scale_factor
    )
    unpacked_latents = unpacked_latents.to(pipeline.vae.dtype)
    latents_mean = (
        torch.tensor(pipeline.vae.config.latents_mean)
        .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
        .to(unpacked_latents.device, unpacked_latents.dtype)
    )
    latents_std = 1.0 / (
        torch.tensor(pipeline.vae.config.latents_std)
        .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
        .to(unpacked_latents.device, unpacked_latents.dtype)
    )
    unpacked_latents = unpacked_latents / latents_std + latents_mean
    image = pipeline.vae.decode(
        unpacked_latents, return_dict=False
    )[0][:, :, 0]
    image = pipeline.image_processor.postprocess(image, output_type=output_type)

    return {
        "images": image,
        "all_latents": all_latents,
        "timesteps": timesteps,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
    }


def predict_velocity(
    transformer,
    pipeline,
    hidden_states,
    timesteps,
    prompt_embeds,
    prompt_embeds_mask,
    negative_prompt_embeds,
    negative_prompt_embeds_mask,
    guidance_scale,
    resolution,
):
    """Qwen-Image transformer forward for packed NFT training latents."""
    batch_size = hidden_states.shape[0]
    img_shapes = [
        [
            (
                1,
                resolution // pipeline.vae_scale_factor // 2,
                resolution // pipeline.vae_scale_factor // 2,
            )
        ]
    ] * batch_size
    do_true_cfg = guidance_scale > 1.0
    normalized_timesteps = timesteps.to(hidden_states.dtype) / 1000.0

    if do_true_cfg:
        model_hidden_states = torch.cat(
            [hidden_states, hidden_states], dim=0
        )
        model_timesteps = torch.cat(
            [normalized_timesteps, normalized_timesteps], dim=0
        )
        model_prompt_embeds = torch.cat(
            [prompt_embeds, negative_prompt_embeds], dim=0
        )
        model_prompt_embeds_mask = torch.cat(
            [prompt_embeds_mask, negative_prompt_embeds_mask], dim=0
        )
        model_img_shapes = img_shapes * 2
    else:
        model_hidden_states = hidden_states
        model_timesteps = normalized_timesteps
        model_prompt_embeds = prompt_embeds
        model_prompt_embeds_mask = prompt_embeds_mask
        model_img_shapes = img_shapes

    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.full(
            [model_hidden_states.shape[0]],
            guidance_scale,
            device=hidden_states.device,
            dtype=torch.float32,
        )
    else:
        guidance = None

    prediction = transformer(
        hidden_states=model_hidden_states,
        timestep=model_timesteps,
        guidance=guidance,
        encoder_hidden_states_mask=model_prompt_embeds_mask,
        encoder_hidden_states=model_prompt_embeds,
        img_shapes=model_img_shapes,
        attention_kwargs=pipeline.attention_kwargs,
        return_dict=False,
    )[0]

    if do_true_cfg:
        conditional_prediction, negative_prediction = prediction.chunk(2, dim=0)
        combined_prediction = negative_prediction + guidance_scale * (
            conditional_prediction - negative_prediction
        )
        conditional_norm = torch.norm(
            conditional_prediction, dim=-1, keepdim=True
        )
        combined_norm = torch.norm(
            combined_prediction, dim=-1, keepdim=True
        )
        prediction = combined_prediction * (
            conditional_norm / combined_norm
        )
    return prediction


def pad_sample_embeddings(samples):
    max_sequence_length = max(
        sample["prompt_embeds_mask"].shape[1] for sample in samples
    )
    for sample in samples:
        sequence_padding = max_sequence_length - sample["prompt_embeds"].shape[1]
        sample["prompt_embeds"] = torch.nn.functional.pad(
            sample["prompt_embeds"], (0, 0, 0, sequence_padding), value=0
        )
        sample["prompt_embeds_mask"] = torch.nn.functional.pad(
            sample["prompt_embeds_mask"], (0, sequence_padding), value=0
        )
        if sample["negative_prompt_embeds"].shape[1] > 0:
            negative_padding = (
                max_sequence_length
                - sample["negative_prompt_embeds"].shape[1]
            )
            sample["negative_prompt_embeds"] = torch.nn.functional.pad(
                sample["negative_prompt_embeds"],
                (0, 0, 0, negative_padding),
                value=0,
            )
            sample["negative_prompt_embeds_mask"] = (
                torch.nn.functional.pad(
                    sample["negative_prompt_embeds_mask"],
                    (0, negative_padding),
                    value=0,
                )
            )


def submit_reward(executor, reward_fn, images, prompts, metadata, only_strict):
    if getattr(reward_fn, "uses_training_device", False):
        return ImmediateResult(
            reward_fn(images, prompts, metadata, only_strict=only_strict)
        )
    reward_future = executor.submit(
        reward_fn, images, prompts, metadata, only_strict=only_strict
    )
    time.sleep(0)
    return reward_future


def eval_model(
    pipeline,
    adapter_controller,
    test_dataloader,
    config,
    rank,
    local_rank,
    world_size,
    device,
    global_step,
    reward_fn,
    executor,
    autocast,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    adapter_controller.set_adapter("default")
    pipeline.transformer.eval()
    all_rewards = defaultdict(list)
    last_batch = None

    for prompts, prompt_metadata in tqdm(
        test_dataloader,
        desc="Eval",
        disable=local_rank != 0,
        position=0,
    ):
        with autocast(), torch.no_grad():
            collected_data = qwenimage_nft_pipeline(
                pipeline,
                prompts,
                negative_prompt=[" "] * len(prompts),
                num_inference_steps=config.sample.eval_num_steps,
                true_cfg_scale=config.sample.eval_guidance_scale,
                output_type="pt",
                height=config.resolution,
                width=config.resolution,
                noise_level=0.0,
                deterministic=True,
                solver="flow",
            )
        images = collected_data["images"]
        reward_future = submit_reward(
            executor,
            reward_fn,
            images,
            prompts,
            prompt_metadata,
            only_strict=False,
        )
        rewards, _ = reward_future.result()
        offload_reward_fn = getattr(reward_fn, "offload", None)
        if offload_reward_fn is not None:
            offload_reward_fn()

        for key, value in rewards.items():
            gathered_value = gather_tensor(
                torch.as_tensor(value, device=device), world_size
            ).cpu().float().numpy()
            all_rewards[key].append(gathered_value)
        last_batch = (images, prompts, rewards)

    if last_batch is None:
        raise RuntimeError("Evaluation dataloader produced no batches.")

    images, prompts, rewards = last_batch
    gathered_images = gather_tensor(
        torch.as_tensor(images, device=device), world_size
    ).cpu().float().numpy()
    gathered_prompts = gather_prompt_strings(prompts, world_size)
    gathered_last_rewards = {
        key: gather_tensor(
            torch.as_tensor(value, device=device), world_size
        ).cpu().float().numpy()
        for key, value in rewards.items()
    }
    all_rewards = {
        key: np.concatenate(values) for key, values in all_rewards.items()
    }

    if rank == 0:
        with tempfile.TemporaryDirectory() as temporary_directory:
            num_samples = min(EVAL_LOG_IMAGE_COUNT, len(gathered_images))
            for index in range(num_samples):
                image = gathered_images[index]
                pil_image = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil_image = pil_image.resize(
                    (config.resolution, config.resolution)
                )
                pil_image.save(
                    os.path.join(temporary_directory, f"{index}.jpg")
                )

            wandb.log(
                {
                    "eval/images": [
                        wandb.Image(
                            os.path.join(
                                temporary_directory, f"{index}.jpg"
                            ),
                            caption=(
                                f"{gathered_prompts[index][:1000]} | "
                                + " | ".join(
                                    f"{key}: {gathered_last_rewards[key][index]:.2f}"
                                    for key in gathered_last_rewards
                                    if gathered_last_rewards[key][index] != -10
                                )
                            ),
                        )
                        for index in range(num_samples)
                    ],
                    **{
                        f"eval/reward_{key}": np.mean(value[value != -10])
                        for key, value in all_rewards.items()
                    },
                },
                step=global_step,
                commit=False,
            )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)


def resolve_resume_checkpoint(config, rank, world_size):
    expected = expected_metadata(
        config,
        trainer=QWEN_NFT_TRAINER_NAME,
        world_size=world_size,
    )
    search_root = config.resume_from if config.resume_from else config.save_dir
    checkpoint_holder = [None]
    if rank == 0:
        checkpoint_holder[0] = find_latest_checkpoint(
            search_root,
            expected,
            include_checkpoints_subdir=True,
        )
    if world_size > 1:
        dist.broadcast_object_list(checkpoint_holder, src=0)
    checkpoint = checkpoint_holder[0]
    if config.resume_from and checkpoint is None:
        raise ValueError(
            f"No compatible complete checkpoint found under {config.resume_from}"
        )
    return checkpoint


def optimizer_state_for_save(optimizer, optimizer_offload_hook):
    state_dict = optimizer.state_dict()
    if optimizer_offload_hook is None:
        return state_dict

    for live_group, saved_group in zip(
        optimizer.param_groups,
        state_dict["param_groups"],
        strict=True,
    ):
        for parameter, parameter_id in zip(
            live_group["params"],
            saved_group["params"],
            strict=True,
        ):
            cpu_state = optimizer_offload_hook.cpu_states.get(parameter)
            if cpu_state is not None:
                state_dict["state"].setdefault(parameter_id, {}).update(cpu_state)
    return state_dict


def adapter_state_for_save(named_adapter_parameters):
    return {
        f"{adapter_name}:{parameter_name}": parameter.detach()
        .cpu()
        .clone()
        .contiguous()
        for adapter_name, named_parameters in named_adapter_parameters.items()
        for parameter_name, parameter in named_parameters
    }


def load_adapter_state(checkpoint_dir, named_adapter_parameters):
    state_path = os.path.join(checkpoint_dir, RESUME_ADAPTERS_NAME)
    with load_safetensors_staged(state_path) as state_dict:
        expected_keys = {
            f"{adapter_name}:{parameter_name}"
            for adapter_name, named_parameters in named_adapter_parameters.items()
            for parameter_name, _ in named_parameters
        }
        if set(state_dict) != expected_keys:
            missing = sorted(expected_keys - set(state_dict))
            unexpected = sorted(set(state_dict) - expected_keys)
            raise ValueError(
                "Checkpoint adapter state mismatch: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        for adapter_name, named_parameters in named_adapter_parameters.items():
            for parameter_name, parameter in named_parameters:
                saved_parameter = state_dict[
                    f"{adapter_name}:{parameter_name}"
                ]
                if saved_parameter.shape != parameter.shape:
                    raise ValueError(
                        f"Checkpoint shape mismatch for {adapter_name}:"
                        f"{parameter_name}: {tuple(saved_parameter.shape)} != "
                        f"{tuple(parameter.shape)}"
                    )
                parameter.data.copy_(
                    saved_parameter.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )


def load_resume_checkpoint(
    checkpoint_dir,
    named_adapter_parameters,
    optimizer,
    optimizer_offload_hook,
    grad_scaler,
    ema,
    stat_tracker,
    config,
    train_sampler,
    rank,
    world_size,
):
    metadata = load_metadata(checkpoint_dir)
    expected = expected_metadata(
        config,
        trainer=QWEN_NFT_TRAINER_NAME,
        world_size=world_size,
    )
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Checkpoint metadata mismatch for {key}: "
                f"checkpoint={metadata.get(key)!r}, current={value!r}"
            )

    load_adapter_state(checkpoint_dir, named_adapter_parameters)
    optimizer_state_path = os.path.join(
        checkpoint_dir,
        RESUME_OPTIMIZER_NAME,
    )
    optimizer.load_state_dict(
        torch_load(optimizer_state_path, map_location="cpu")
    )
    if optimizer_offload_hook is not None:
        optimizer_offload_hook.post_step_hook(optimizer, (), {})

    if config.train.ema and ema is not None:
        ema.load_state_dict(
            torch_load(
                os.path.join(checkpoint_dir, RESUME_EMA_NAME),
                map_location="cpu",
            )
        )

    rank_state = torch_load(
        os.path.join(
            checkpoint_dir,
            f"{RESUME_TRAINING_STATE_NAME}.{rank}",
        ),
        map_location="cpu",
    )
    if rank_state["rank"] != rank:
        raise ValueError(f"Checkpoint rank mismatch for rank {rank}")
    grad_scaler.load_state_dict(rank_state["grad_scaler"])
    stat_tracker.load_state_dict(rank_state.get("stat_tracker", {}))
    train_sampler.set_epoch(rank_state["sampler_epoch"])
    load_rng_state_dict(rank_state["rng_state"])
    if dist.is_initialized():
        dist.barrier()
    return rank_state["epoch"], rank_state["global_step"]


def save_checkpoint(
    config,
    adapter_controller,
    optimizer,
    optimizer_offload_hook,
    grad_scaler,
    ema,
    stat_tracker,
    transformer_trainable_parameters,
    named_adapter_parameters,
    train_sampler,
    epoch,
    global_step,
    rank,
    world_size,
):
    save_root = os.path.join(
        config.save_dir,
        f"checkpoint-{epoch}",
    )
    temporary_root = os.path.join(
        config.save_dir,
        f".checkpoint-{epoch}.tmp",
    )
    checkpoint_exists = [False]
    if rank == 0:
        checkpoint_exists[0] = os.path.exists(save_root)
    if world_size > 1:
        dist.broadcast_object_list(checkpoint_exists, src=0)
    if checkpoint_exists[0]:
        raise ValueError(
            f"Checkpoint already exists and will not be overwritten: {save_root}"
        )

    if rank == 0:
        if os.path.exists(temporary_root):
            shutil.rmtree(temporary_root)
        os.makedirs(config.save_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    os.makedirs(temporary_root, exist_ok=True)

    # The ignored LoRA leaves, their optimizer state, and EMA are replicated
    # across ranks; save one shared copy while keeping RNG/scaler state per rank.
    if rank == 0:
        save_safetensors_staged(
            adapter_state_for_save(named_adapter_parameters),
            os.path.join(temporary_root, RESUME_ADAPTERS_NAME),
            metadata={"format": "pt"},
        )
        torch.save(
            optimizer_state_for_save(optimizer, optimizer_offload_hook),
            os.path.join(temporary_root, RESUME_OPTIMIZER_NAME),
        )
        if config.train.ema and ema is not None:
            torch.save(
                ema.state_dict(),
                os.path.join(temporary_root, RESUME_EMA_NAME),
            )
    rank_state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "rank": int(rank),
        "rng_state": rng_state_dict(),
        "sampler_epoch": int(train_sampler.epoch),
        "grad_scaler": grad_scaler.state_dict(),
        "stat_tracker": stat_tracker.state_dict(),
    }
    torch.save(
        rank_state,
        os.path.join(
            temporary_root,
            f"{RESUME_TRAINING_STATE_NAME}.{rank}",
        ),
    )
    if dist.is_initialized():
        dist.barrier()

    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    if rank == 0:
        lora_root = os.path.join(temporary_root, "lora")
        os.makedirs(lora_root, exist_ok=True)
        raw_state_dict = {
            name: parameter.detach().cpu().clone()
            for name, parameter in named_adapter_parameters["default"]
        }
        adapter_state_dict = get_peft_model_state_dict(
            adapter_controller,
            state_dict=raw_state_dict,
            adapter_name="default",
        )
        save_safetensors_staged(
            {
                name: tensor.contiguous()
                for name, tensor in adapter_state_dict.items()
            },
            os.path.join(lora_root, "adapter_model.safetensors"),
            metadata={"format": "pt"},
        )
        adapter_controller.peft_config["default"].save_pretrained(lora_root)
    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        metadata = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "sampler_epoch": int(train_sampler.epoch),
            **expected_metadata(
                config,
                trainer=QWEN_NFT_TRAINER_NAME,
                world_size=world_size,
            ),
        }
        with open(
            os.path.join(temporary_root, META_NAME),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
        with open(
            os.path.join(temporary_root, COMPLETE_NAME),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {"epoch": int(epoch), "global_step": int(global_step)},
                file,
                sort_keys=True,
            )
        os.replace(temporary_root, save_root)

        if config.num_checkpoint_limit > 0:
            checkpoint_directories = []
            for entry in os.listdir(config.save_dir):
                checkpoint_path = os.path.join(config.save_dir, entry)
                if not entry.startswith("checkpoint-"):
                    continue
                try:
                    checkpoint_step = int(entry.removeprefix("checkpoint-"))
                except ValueError:
                    continue
                checkpoint_directories.append(
                    (checkpoint_step, checkpoint_path)
                )
            checkpoint_directories.sort()
            for _, checkpoint_directory in checkpoint_directories[
                : -config.num_checkpoint_limit
            ]:
                shutil.rmtree(checkpoint_directory)
    if dist.is_initialized():
        dist.barrier()


def main(_):
    config = FLAGS.config
    _apply_fix_config(config)
    if config.loss_type not in ("nft", "awm"):
        raise ValueError(f"Unknown loss_type: {config.loss_type}")
    if config.train.adv_mode not in (
        "all",
        "positive_only",
        "negative_only",
        "one_only",
        "binary",
    ):
        raise ValueError(f"Unknown train.adv_mode: {config.train.adv_mode}")
    if config.train.ema_beta > 0 and config.train.kl_ema_weight != "Uniform":
        raise ValueError(
            f"Unknown train.kl_ema_weight: {config.train.kl_ema_weight}"
        )
    if not config.use_lora:
        raise ValueError("NFT requires use_lora=True for the dual-adapter policy.")
    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False

    is_distributed, rank, world_size, local_rank = init_distributed()
    device = (
        torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    resume_checkpoint = resolve_resume_checkpoint(config, rank, world_size)

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
    if config.loss_type != "nft":
        config.run_name += f"_{config.loss_type}"

    timestep_selection = (
        config.train.timestep_selection
        if "timestep_selection" in config.train
        else "random"
    )
    if timestep_selection == "fixed":
        if not (
            0
            <= config.start_timesteps
            < config.end_timesteps
            <= config.sample.num_steps
        ):
            raise ValueError(
                "Fixed timestep range must satisfy "
                "0 <= start_timesteps < end_timesteps <= sample.num_steps."
            )
        num_train_timesteps = (
            config.end_timesteps - config.start_timesteps
        )
    elif timestep_selection == "random":
        num_train_timesteps = int(
            config.sample.num_steps * config.train.timestep_fraction
        )
    else:
        raise ValueError(
            f"Unknown train.timestep_selection: {timestep_selection}"
        )
    if not 0 < num_train_timesteps <= config.sample.num_steps:
        raise ValueError(
            "Timestep selection produced an invalid timestep count: "
            f"{num_train_timesteps}"
        )

    project_dir = os.path.join(config.logdir, config.run_name)
    os.makedirs(project_dir, exist_ok=True)
    if rank == 0:
        wandb.init(
            project="diffusion-rl",
            name=config.run_name,
            config=config.to_dict(),
        )
        logger.info("\n%s", config)

    set_seed(config.seed, device_specific=True)
    inference_dtype = torch.float32
    if config.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline = DiffusionPipeline.from_pretrained(
        config.pretrained.model, torch_dtype=inference_dtype
    )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=local_rank != 0,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    pipeline.vae.to(device, dtype=torch.float32)

    target_modules = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "img_mlp.net.0.proj",
        "img_mlp.net.2",
        "txt_mlp.net.0.proj",
        "txt_mlp.net.2",
    ]
    transformer_lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    if config.train.lora_path:
        pipeline.transformer = PeftModel.from_pretrained(
            pipeline.transformer,
            config.train.lora_path,
            is_trainable=True,
        )
        pipeline.transformer.set_adapter("default")
    else:
        pipeline.transformer = get_peft_model(
            pipeline.transformer, transformer_lora_config
        )

    adapter_controller = pipeline.transformer
    adapter_controller.add_adapter("old", transformer_lora_config)
    if config.train.ema_beta > 0:
        adapter_controller.add_adapter("ema", transformer_lora_config)

    transformer_trainable_parameters = collect_trainable_adapter_parameters(
        adapter_controller, "default"
    )
    named_transformer_trainable_parameters = (
        collect_named_adapter_parameters(
            adapter_controller, transformer_trainable_parameters
        )
    )
    old_transformer_trainable_parameters = (
        collect_trainable_adapter_parameters(adapter_controller, "old")
    )
    named_old_transformer_trainable_parameters = (
        collect_named_adapter_parameters(
            adapter_controller,
            old_transformer_trainable_parameters,
        )
    )
    validate_parameter_pairs(
        transformer_trainable_parameters,
        old_transformer_trainable_parameters,
        "old",
    )
    ema_transformer_trainable_parameters = None
    named_ema_transformer_trainable_parameters = None
    if config.train.ema_beta > 0:
        ema_transformer_trainable_parameters = (
            collect_trainable_adapter_parameters(adapter_controller, "ema")
        )
        named_ema_transformer_trainable_parameters = (
            collect_named_adapter_parameters(
                adapter_controller,
                ema_transformer_trainable_parameters,
            )
        )
        validate_parameter_pairs(
            transformer_trainable_parameters,
            ema_transformer_trainable_parameters,
            "ema",
        )
    named_adapter_parameters = {
        "default": named_transformer_trainable_parameters,
        "old": named_old_transformer_trainable_parameters,
    }
    if named_ema_transformer_trainable_parameters is not None:
        named_adapter_parameters["ema"] = (
            named_ema_transformer_trainable_parameters
        )

    copy_adapter_parameters(
        transformer_trainable_parameters,
        old_transformer_trainable_parameters,
    )
    if ema_transformer_trainable_parameters is not None:
        copy_adapter_parameters(
            transformer_trainable_parameters,
            ema_transformer_trainable_parameters,
        )
    adapter_controller.set_adapter("default")
    ignored_lora_modules = collect_lora_leaf_modules(adapter_controller)

    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        cpu_offload=False,
        num_replicate=1,
        num_shard=world_size,
        mixed_precision_dtype=inference_dtype,
        use_activation_checkpointing=config.activation_checkpointing,
        use_device_mesh=False,
    )
    transformer = adapter_controller.cpu().to(dtype=torch.float32)
    transformer = fsdp_wrapper(
        transformer,
        fsdp_config,
        get_transformer_layer_cls,
        ignored_modules=ignored_lora_modules,
    )
    # FSDP does not own ignored parameters. Move the replicated LoRA leaves
    # explicitly so their optimizer state and adapter blends stay on this rank.
    for lora_module in ignored_lora_modules:
        lora_module.to(device)
    pipeline.transformer = transformer

    pipeline.text_encoder.cpu().to(dtype=torch.float32)
    pipeline.text_encoder = fsdp_wrapper(
        pipeline.text_encoder, fsdp_config, get_transformer_layer_cls
    )

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    ema = (
        EMAModuleWrapper(
            transformer_trainable_parameters,
            decay=0.9,
            update_step_interval=1,
            device=device,
        )
        if config.train.ema
        else None
    )

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam."
            ) from error
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    optimizer_offload_hook = None
    if config.fsdp_optimizer_offload:
        _, optimizer_offload_hook = register_optimizer_offload_hooks(optimizer)

    reward_fn = flow_grpo.rewards.multi_score(
        device,
        config.reward_fn,
        config.reward_backends,
        hpsv3_offload=config.hpsv3_offload,
    )
    eval_reward_fn = reward_fn

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
        collate_fn = TextPromptDataset.collate_fn
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
        collate_fn = GenevalPromptDataset.collate_fn
    else:
        raise NotImplementedError(
            f"Unsupported prompt_fn for dataset training: {config.prompt_fn}"
        )

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    train_dataloader_generator = torch.Generator().manual_seed(
        config.seed + rank
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        generator=train_dataloader_generator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        ),
        batch_size=config.sample.test_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=1,
    )

    if not config.per_prompt_stat_tracking:
        raise ValueError("NFT requires per_prompt_stat_tracking=True.")
    stat_tracker = PerPromptStatTracker(config.sample.global_std)

    if config.mixed_precision == "fp16":
        autocast = lambda: torch.cuda.amp.autocast(dtype=torch.float16)
    elif config.mixed_precision == "bf16":
        autocast = lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext
    grad_scaler = ShardedGradScaler(
        enabled=config.mixed_precision == "fp16"
    )

    executor = futures.ThreadPoolExecutor(max_workers=8)
    samples_per_epoch = (
        config.sample.train_batch_size
        * world_size
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * world_size
        * config.train.gradient_accumulation_steps
    )
    logger.info("***** Running Qwen-Image NFT/AWM training *****")
    logger.info(
        "  Sample batch size per device = %s",
        config.sample.train_batch_size,
    )
    logger.info("  Train batch size per device = %s", config.train.batch_size)
    logger.info(
        "  Gradient accumulation steps = %s",
        config.train.gradient_accumulation_steps,
    )
    logger.info("  Total samples per epoch = %s", samples_per_epoch)
    logger.info("  Total train batch size = %s", total_train_batch_size)
    logger.info("  Number of inner epochs = %s", config.train.num_inner_epochs)
    logger.info(
        "  Solver = %s, deterministic = %s",
        config.sample.solver,
        config.sample.deterministic,
    )

    epoch = 0
    global_step = 0
    if resume_checkpoint is not None:
        epoch, global_step = load_resume_checkpoint(
            resume_checkpoint,
            named_adapter_parameters,
            optimizer,
            optimizer_offload_hook,
            grad_scaler,
            ema,
            stat_tracker,
            config,
            train_sampler,
            rank,
            world_size,
        )
        logger.info(
            "Resumed NFT training from %s at epoch=%s, global_step=%s",
            resume_checkpoint,
            epoch,
            global_step,
        )
    last_checkpoint_epoch = epoch if resume_checkpoint is not None else -1
    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    while epoch < config.num_epochs:
        adapter_controller.set_adapter("default")
        pipeline.transformer.eval()
        if config.eval and epoch % config.eval_freq == 0:
            eval_model(
                pipeline,
                adapter_controller,
                test_dataloader,
                config,
                rank,
                local_rank,
                world_size,
                device,
                global_step,
                eval_reward_fn,
                executor,
                autocast,
                ema,
                transformer_trainable_parameters,
            )

        adapter_controller.set_adapter("old")
        pipeline.transformer.eval()
        sample_list = []
        epoch_prompts = []
        last_prompts = None
        last_images = None
        for batch_index in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=local_rank != 0,
            position=0,
        ):
            sampler_epoch = (
                epoch * config.sample.num_batches_per_epoch + batch_index
            )
            train_sampler.set_epoch(sampler_epoch)
            prompts, prompt_metadata = next(train_iter)
            epoch_prompts.extend(prompts)
            generator = (
                create_generator(
                    prompts, base_seed=epoch * 10000 + batch_index
                )
                if config.sample.same_latent
                else None
            )

            with autocast(), torch.no_grad():
                collected_data = qwenimage_nft_pipeline(
                    pipeline,
                    prompts,
                    negative_prompt=[" "] * len(prompts),
                    num_inference_steps=config.sample.num_steps,
                    true_cfg_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    generator=generator,
                    deterministic=config.sample.deterministic,
                    solver=config.sample.solver,
                )

            images = collected_data["images"]
            latents_clean = collected_data["all_latents"][-1]
            timesteps = collected_data["timesteps"].unsqueeze(0).repeat(
                len(prompts), 1
            )
            next_timesteps = torch.cat(
                [timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])],
                dim=1,
            )
            reward_future = submit_reward(
                executor,
                reward_fn,
                images,
                prompts,
                prompt_metadata,
                only_strict=True,
            )
            sample_list.append(
                {
                    "prompt_embeds": collected_data["prompt_embeds"],
                    "prompt_embeds_mask": collected_data[
                        "prompt_embeds_mask"
                    ],
                    "negative_prompt_embeds": collected_data[
                        "negative_prompt_embeds"
                    ],
                    "negative_prompt_embeds_mask": collected_data[
                        "negative_prompt_embeds_mask"
                    ],
                    "timesteps": timesteps,
                    "next_timesteps": next_timesteps,
                    "latents_clean": latents_clean,
                    "rewards": reward_future,
                }
            )
            last_prompts = prompts
            last_images = images

        pad_sample_embeddings(sample_list)
        last_rewards = None
        for sample in tqdm(
            sample_list,
            desc="Waiting for rewards",
            disable=local_rank != 0,
            position=0,
        ):
            rewards, _ = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=device).float()
                for key, value in rewards.items()
            }
            last_rewards = sample["rewards"]
        offload_reward_fn = getattr(reward_fn, "offload", None)
        if offload_reward_fn is not None:
            offload_reward_fn()

        samples = {
            key: (
                torch.cat([sample[key] for sample in sample_list], dim=0)
                if not isinstance(sample_list[0][key], dict)
                else {
                    sub_key: torch.cat(
                        [
                            sample[key][sub_key]
                            for sample in sample_list
                        ],
                        dim=0,
                    )
                    for sub_key in sample_list[0][key]
                }
            )
            for key in sample_list[0]
        }

        if epoch % 10 == 0 and rank == 0:
            with tempfile.TemporaryDirectory() as temporary_directory:
                num_samples = min(15, len(last_images))
                for index in range(num_samples):
                    image = last_images[index].cpu().float().numpy()
                    pil_image = Image.fromarray(
                        (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil_image = pil_image.resize(
                        (config.resolution, config.resolution)
                    )
                    pil_image.save(
                        os.path.join(temporary_directory, f"{index}.jpg")
                    )
                wandb.log(
                    {
                        "train/images": [
                            wandb.Image(
                                os.path.join(
                                    temporary_directory, f"{index}.jpg"
                                ),
                                caption=(
                                    f"{last_prompts[index][:100]} | "
                                    f"avg: {last_rewards['avg'][index]:.2f}"
                                ),
                            )
                            for index in range(num_samples)
                        ]
                    },
                    step=global_step,
                    commit=False,
                )

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"].clone()
        samples["rewards"]["avg"] = (
            samples["rewards"]["avg"]
            .unsqueeze(1)
            .repeat(1, num_train_timesteps)
        )
        gathered_rewards = {
            key: gather_tensor(value, world_size).cpu().float().numpy()
            for key, value in samples["rewards"].items()
        }
        if rank == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward/{key}": value.mean()
                        for key, value in gathered_rewards.items()
                        if "_strict_accuracy" not in key
                        and "_accuracy" not in key
                    },
                },
                step=global_step,
                commit=False,
            )

        prompts_all = gather_prompt_strings(epoch_prompts, world_size)
        advantages = stat_tracker.update(
            prompts_all, gathered_rewards["avg"]
        )
        group_size, trained_prompt_num = stat_tracker.get_stats()
        zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(
            prompts_all, gathered_rewards
        )
        if rank == 0:
            wandb.log(
                {
                    "stats/group_size": group_size,
                    "stats/trained_prompt_num": trained_prompt_num,
                    "stats/zero_std_ratio": zero_std_ratio,
                    "stats/reward_std_mean": reward_std_mean,
                    "stats/mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                    "stats/mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                    "stats/mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                    "stats/mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                    "stats/mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                },
                step=global_step,
                commit=False,
            )
        stat_tracker.clear()

        advantages = torch.as_tensor(advantages, dtype=torch.float32)
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        samples_per_rank = samples["timesteps"].shape[0]
        if advantages.shape[0] != world_size * samples_per_rank:
            raise RuntimeError(
                "Gathered advantage count does not match per-rank samples."
            )
        samples["advantages"] = advantages.reshape(
            world_size, samples_per_rank, -1
        )[rank].to(device=device, dtype=torch.float32)
        del samples["rewards"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        if num_timesteps != config.sample.num_steps:
            raise RuntimeError(
                f"Expected {config.sample.num_steps} timesteps, got "
                f"{num_timesteps}."
            )
        if total_batch_size % config.sample.num_batches_per_epoch != 0:
            raise RuntimeError(
                "Collected samples cannot be evenly rebatched for training."
            )

        adapter_controller.set_adapter("default")
        pipeline.transformer.train()
        accumulation_steps = (
            config.train.gradient_accumulation_steps * num_train_timesteps
        )
        accumulation_index = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            sample_permutation = create_train_permutation(
                total_batch_size,
                config.seed,
                epoch,
                inner_epoch,
                config.train.num_inner_epochs,
                device,
            )
            shuffled = {
                key: value[sample_permutation]
                for key, value in samples.items()
            }
            if timestep_selection == "random":
                timestep_permutations = torch.stack(
                    [
                        torch.randperm(num_timesteps, device=device)
                        for _ in range(total_batch_size)
                    ]
                )
                batch_indices = torch.arange(
                    total_batch_size, device=device
                )[:, None]
                for key in ("timesteps", "next_timesteps"):
                    shuffled[key] = shuffled[key][
                        batch_indices, timestep_permutations
                    ]
            else:
                for key in ("timesteps", "next_timesteps"):
                    shuffled[key] = shuffled[key][
                        :,
                        config.start_timesteps : config.end_timesteps,
                    ]

            training_batch_size = (
                total_batch_size // config.sample.num_batches_per_epoch
            )
            samples_batched = []
            for batch_index in range(
                config.sample.num_batches_per_epoch
            ):
                start = batch_index * training_batch_size
                end = start + training_batch_size
                samples_batched.append(
                    {
                        key: value[start:end]
                        for key, value in shuffled.items()
                    }
                )

            info = defaultdict(list)
            for batch_index, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=local_rank != 0,
            ):
                for timestep_index in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=local_rank != 0,
                ):
                    accumulation_index += 1
                    should_sync = (
                        accumulation_index % accumulation_steps == 0
                    )
                    with autocast():
                        x0 = sample["latents_clean"]
                        timestep = sample["timesteps"][:, timestep_index]
                        t = timestep.float() / 1000.0
                        t_expanded = t.view(
                            -1, *([1] * (x0.ndim - 1))
                        )
                        noise = torch.randn_like(x0.float())
                        xt = (1.0 - t_expanded) * x0 + t_expanded * noise

                        adapter_controller.set_adapter("old")
                        with torch.no_grad():
                            old_prediction = predict_velocity(
                                transformer,
                                pipeline,
                                xt,
                                timestep,
                                sample["prompt_embeds"],
                                sample["prompt_embeds_mask"],
                                sample["negative_prompt_embeds"],
                                sample["negative_prompt_embeds_mask"],
                                config.sample.guidance_scale,
                                config.resolution,
                            ).detach()

                        adapter_controller.set_adapter("default")
                        forward_prediction = predict_velocity(
                            transformer,
                            pipeline,
                            xt,
                            timestep,
                            sample["prompt_embeds"],
                            sample["prompt_embeds_mask"],
                            sample["negative_prompt_embeds"],
                            sample["negative_prompt_embeds_mask"],
                            config.sample.guidance_scale,
                            config.resolution,
                        )

                        with torch.no_grad(), adapter_controller.disable_adapter():
                            ref_forward_prediction = predict_velocity(
                                transformer,
                                pipeline,
                                xt,
                                timestep,
                                sample["prompt_embeds"],
                                sample["prompt_embeds_mask"],
                                sample["negative_prompt_embeds"],
                                sample["negative_prompt_embeds_mask"],
                                config.sample.guidance_scale,
                                config.resolution,
                            )
                        adapter_controller.set_adapter("default")

                        ema_prediction = None
                        if config.train.ema_beta > 0:
                            adapter_controller.set_adapter("ema")
                            with torch.no_grad():
                                ema_prediction = predict_velocity(
                                    transformer,
                                    pipeline,
                                    xt,
                                    timestep,
                                    sample["prompt_embeds"],
                                    sample["prompt_embeds_mask"],
                                    sample["negative_prompt_embeds"],
                                    sample[
                                        "negative_prompt_embeds_mask"
                                    ],
                                    config.sample.guidance_scale,
                                    config.resolution,
                                )
                            adapter_controller.set_adapter("default")

                    advantages_clip = torch.clamp(
                        sample["advantages"][:, timestep_index],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    if config.train.adv_mode == "positive_only":
                        advantages_clip = torch.clamp(
                            advantages_clip,
                            0,
                            config.train.adv_clip_max,
                        )
                    elif config.train.adv_mode == "negative_only":
                        advantages_clip = torch.clamp(
                            advantages_clip,
                            -config.train.adv_clip_max,
                            0,
                        )
                    elif config.train.adv_mode == "one_only":
                        advantages_clip = torch.where(
                            advantages_clip > 0,
                            torch.ones_like(advantages_clip),
                            torch.zeros_like(advantages_clip),
                        )
                    elif config.train.adv_mode == "binary":
                        advantages_clip = torch.sign(advantages_clip)

                    spatial_dims = tuple(range(1, x0.ndim))
                    loss_terms = {
                        "x0_norm": torch.mean(x0**2).detach(),
                        "x0_norm_max": torch.max(x0**2).detach(),
                        "old_deviate": torch.mean(
                            (forward_prediction - old_prediction) ** 2
                        ).detach(),
                        "old_deviate_max": torch.max(
                            (forward_prediction - old_prediction) ** 2
                        ).detach(),
                    }

                    if config.loss_type == "nft":
                        normalized_advantage = (
                            advantages_clip
                            / config.train.adv_clip_max
                            / 2.0
                            + 0.5
                        )
                        reward_weight = torch.clamp(
                            normalized_advantage, 0, 1
                        )
                        positive_prediction = (
                            config.beta * forward_prediction
                            + (1.0 - config.beta)
                            * old_prediction.detach()
                        )
                        implicit_negative_prediction = (
                            (1.0 + config.beta)
                            * old_prediction.detach()
                            - config.beta * forward_prediction
                        )

                        x0_prediction = (
                            xt - t_expanded * positive_prediction
                        )
                        with torch.no_grad():
                            weight_factor = (
                                torch.abs(
                                    x0_prediction.double() - x0.double()
                                )
                                .mean(
                                    dim=spatial_dims,
                                    keepdim=True,
                                )
                                .clip(min=0.00001)
                                .to(dtype=x0_prediction.dtype)
                            )
                        positive_loss = (
                            (x0_prediction - x0) ** 2 / weight_factor
                        ).mean(dim=spatial_dims)

                        negative_x0_prediction = (
                            xt
                            - t_expanded
                            * implicit_negative_prediction
                        )
                        with torch.no_grad():
                            negative_weight_factor = (
                                torch.abs(
                                    negative_x0_prediction.double()
                                    - x0.double()
                                )
                                .mean(
                                    dim=spatial_dims,
                                    keepdim=True,
                                )
                                .clip(min=0.00001)
                                .to(
                                    dtype=negative_x0_prediction.dtype
                                )
                            )
                        negative_loss = (
                            (negative_x0_prediction - x0) ** 2
                            / negative_weight_factor
                        ).mean(dim=spatial_dims)
                        original_policy_loss = (
                            reward_weight
                            * positive_loss
                            / config.beta
                            + (1.0 - reward_weight)
                            * negative_loss
                            / config.beta
                        )
                        policy_loss = (
                            original_policy_loss
                            * config.train.adv_clip_max
                        ).mean()
                    else:
                        instantaneous_velocity = noise - x0
                        advantage_weight = advantages_clip.view(
                            -1, *([1] * (x0.ndim - 1))
                        )
                        original_policy_loss = (
                            advantage_weight
                            * (
                                forward_prediction
                                - instantaneous_velocity
                            )
                            ** 2
                        ).mean(dim=spatial_dims)
                        policy_loss = original_policy_loss.mean()

                    loss = policy_loss
                    kl_divergence = (
                        (
                            forward_prediction
                            - ref_forward_prediction
                        )
                        ** 2
                    ).mean(dim=spatial_dims)
                    loss = loss + config.train.beta * torch.mean(
                        kl_divergence
                    )
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = (
                        original_policy_loss.mean().detach()
                    )
                    loss_terms["kl_div_loss"] = torch.mean(
                        kl_divergence
                    ).detach()
                    loss_terms["kl_div"] = torch.mean(
                        kl_divergence
                    ).detach()
                    loss_terms["old_kl_div"] = torch.mean(
                        (
                            (
                                old_prediction
                                - ref_forward_prediction
                            )
                            ** 2
                        ).mean(dim=spatial_dims)
                    ).detach()
                    if config.train.ema_beta > 0:
                        ema_kl_divergence = (
                            (forward_prediction - ema_prediction) ** 2
                        ).mean(dim=spatial_dims)
                        loss = loss + config.train.ema_beta * torch.mean(
                            ema_kl_divergence
                        )
                        loss_terms["ema_kl_div_loss"] = torch.mean(
                            ema_kl_divergence
                        ).detach()
                        loss_terms["ema_kl_div"] = torch.mean(
                            ema_kl_divergence
                        ).detach()
                    loss_terms["total_loss"] = loss.detach()

                    for key, value in loss_terms.items():
                        info[key].append(value)
                    grad_scaler.scale(
                        loss / accumulation_steps
                    ).backward()

                    if should_sync:
                        synchronize_replicated_gradients(
                            transformer_trainable_parameters,
                            world_size,
                        )
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            transformer_trainable_parameters,
                            config.train.max_grad_norm,
                        )
                        scale_before_step = grad_scaler.get_scale()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        optimizer_step_skipped = (
                            grad_scaler.is_enabled()
                            and grad_scaler.get_scale() < scale_before_step
                        )
                        optimizer.zero_grad()

                        reduced_info = {
                            key: torch.mean(torch.stack(values))
                            for key, values in info.items()
                        }
                        if is_distributed:
                            for key, value in reduced_info.items():
                                dist.all_reduce(
                                    value, op=dist.ReduceOp.SUM
                                )
                                reduced_info[key] = value / world_size
                        reduced_info.update(
                            {
                                "epoch": epoch,
                                "inner_epoch": inner_epoch,
                            }
                        )
                        if rank == 0:
                            wandb.log(
                                {
                                    **{
                                        f"train/{key}": (
                                            value.item()
                                            if isinstance(
                                                value, torch.Tensor
                                            )
                                            else value
                                        )
                                        for key, value in reduced_info.items()
                                    },
                                    "train/learning_rate": optimizer.param_groups[
                                        0
                                    ]["lr"],
                                },
                                step=global_step,
                                commit=True,
                            )
                        if (
                            not optimizer_step_skipped
                            and config.train.ema
                            and ema is not None
                        ):
                            ema.step(
                                transformer_trainable_parameters,
                                global_step,
                            )
                        if not optimizer_step_skipped:
                            global_step += 1
                        info = defaultdict(list)

        if accumulation_index % accumulation_steps != 0:
            raise RuntimeError(
                "An epoch ended with pending accumulated gradients; adjust "
                "train.gradient_accumulation_steps."
            )
        if is_distributed:
            dist.barrier()

        decay = return_decay(global_step, config.decay_type)
        blend_adapter_parameters(
            transformer_trainable_parameters,
            old_transformer_trainable_parameters,
            decay,
        )
        ema_decay = None
        if config.train.ema_beta > 0:
            if config.train.kl_ema_decay_type == "constant":
                ema_decay = config.train.kl_ema_decay
            elif config.train.kl_ema_decay_type == "linear":
                ema_decay = min(
                    config.train.kl_ema_decay, 0.001 * global_step
                )
            else:
                raise ValueError(
                    "Unknown train.kl_ema_decay_type: "
                    f"{config.train.kl_ema_decay_type}"
                )
            blend_adapter_parameters(
                transformer_trainable_parameters,
                ema_transformer_trainable_parameters,
                ema_decay,
            )
        if rank == 0:
            log_data = {"train/old_policy_decay": decay}
            if ema_decay is not None:
                log_data["train/ema_kl_decay"] = ema_decay
            wandb.log(
                log_data,
                step=global_step,
                commit=False,
            )
        epoch += 1
        if (
            config.save
            and epoch % config.save_freq == 0
            and epoch > last_checkpoint_epoch
        ):
            save_checkpoint(
                config,
                adapter_controller,
                optimizer,
                optimizer_offload_hook,
                grad_scaler,
                ema,
                stat_tracker,
                transformer_trainable_parameters,
                named_adapter_parameters,
                train_sampler,
                epoch,
                global_step,
                rank,
                world_size,
            )
            last_checkpoint_epoch = epoch

    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
