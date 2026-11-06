from collections import defaultdict
import contextlib
import copy
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import datetime
import math
import shutil
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from ml_collections import config_flags
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.utils.data.distributed import DistributedSampler
import logging

logger = logging.getLogger(__name__)
from diffusers import DiffusionPipeline, QwenImageTransformer2DModel
from diffusers.utils.torch_utils import is_compiled_module

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
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.qwenimage_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from flow_grpo.logging_utils import ExperimentLogger

wandb = ExperimentLogger()
from safetensors.torch import load_file, save_file
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

EVAL_PROMPT_COUNT = 2048
EVAL_LOG_IMAGE_COUNT = 30


class ImmediateResult:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


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
        torch.cuda.manual_seed_all(seed + dist.get_rank() if dist.is_initialized() else seed)


def get_timestep_config_value(value, step_index):
    if isinstance(value, (list, tuple)):
        return value[step_index]
    return value


def compute_kde_estimator(
    local_prompt_group_ids,
    local_latents,
    local_velocities,
    local_timesteps,
    global_prompt_group_ids,
    global_clean_latents,
    global_advantages,
    adv_clip_max,
    step_indices,
    kde_bandwidth,
):
    assert kde_bandwidth > 0, "kde_bandwidth must be positive"
    estimator = torch.zeros_like(local_velocities)
    local_batch_size = local_latents.shape[0]
    num_kde_timesteps = len(step_indices)
    stats = {
        "weight_entropy": torch.zeros(
            (local_batch_size, num_kde_timesteps),
            device=local_latents.device,
            dtype=torch.float32,
        ),
        "logits_mean": torch.zeros(
            (local_batch_size, num_kde_timesteps),
            device=local_latents.device,
            dtype=torch.float32,
        ),
        "logits_max": torch.zeros(
            (local_batch_size, num_kde_timesteps),
            device=local_latents.device,
            dtype=torch.float32,
        ),
        "logits_min": torch.zeros(
            (local_batch_size, num_kde_timesteps),
            device=local_latents.device,
            dtype=torch.float32,
        ),
    }
    global_clean_latents = global_clean_latents.float()
    global_advantages = torch.clamp(global_advantages.float(), -adv_clip_max, adv_clip_max)

    for local_index in range(local_batch_size):
        prompt_mask = global_prompt_group_ids == local_prompt_group_ids[local_index]
        clean_latents = global_clean_latents[prompt_mask]
        advantages = global_advantages[prompt_mask]

        for stats_index, step_index in enumerate(step_indices):
            t = local_timesteps[local_index, step_index].float() / 1000.0
            y_t = local_latents[local_index, step_index].float()
            v_base = local_velocities[local_index, step_index].float()
            shifted_clean_latents = (1.0 - t) * clean_latents
            scaled_logits = -(
                (y_t.unsqueeze(0) - shifted_clean_latents)
                .pow(2)
                .flatten(1)
                .sum(dim=1)
            ) / (2.0 * t.pow(2) * kde_bandwidth)
            weights = torch.softmax(scaled_logits, dim=0)
            stats["weight_entropy"][local_index, stats_index] = -(
                weights * torch.log(weights + 1e-8)
            ).sum()
            stats["logits_mean"][local_index, stats_index] = scaled_logits.mean()
            target_velocities = (y_t.unsqueeze(0) - clean_latents) / t
            broadcast_shape = (-1,) + (1,) * (clean_latents.ndim - 1)
            kde_value_grad = (
                weights.view(broadcast_shape)
                * advantages[:, step_index].view(broadcast_shape)
                * (target_velocities - v_base.unsqueeze(0))
            ).sum(dim=0)
            estimator[local_index, step_index] = kde_value_grad.to(estimator.dtype)

    return estimator, stats


def get_policy_gradient_weight(t, config, weight_type, t_j_max=1.0, sqrt_dt=1.0):
    if weight_type == "sto_one_sample":
        return 1.0 + config.sample.noise_level**2 / 2
    elif weight_type == "det_one_sample":
        return -2.0
    elif weight_type in ("det_t_exp_a", "t_exp_a"):
        return -2.0 * (t ** config.weight_exp)
    elif weight_type == "sto_uniform":
        return torch.sqrt(t / (1 - torch.where(t == 1, t_j_max, t))) * config.sample.noise_level / sqrt_dt
    elif weight_type == "sto_t_exp_a":
        return torch.sqrt(t / (1 - torch.where(t == 1, t_j_max, t))) * config.sample.noise_level / sqrt_dt * (t ** config.weight_exp)
    elif weight_type == "vanilla_var_reduction":
        return -(1 + config.sample.noise_level**2 / 2) * sqrt_dt**2 * (1 - torch.where(t == 1, t_j_max, t)) / t
    elif weight_type == "grpo_guard":
        assert sqrt_dt > 0, "invalid sqrt_dt"
        return (1 + config.sample.noise_level**2 / 2) * config.sample.noise_level * torch.sqrt(t / (1 - torch.where(t == 1, t_j_max, t))) / sqrt_dt
    else:
        raise NotImplementedError(f"Unknown weight type: {weight_type}")


def get_off_policy_weight(t, advantage, config, weight_type, t_j_max=1.0, sqrt_dt=1.0):
    if weight_type in ("sto_one_sample", "vanilla_var_reduction"):
        w = (1 + config.sample.noise_level**2 / 2)**2 / (2 * config.sample.noise_level**2)
        return advantage * w * sqrt_dt**2 * (1 - torch.where(t == 1, t_j_max, t)) / t
    elif weight_type == "det_one_sample":
        return advantage
    elif weight_type == "sto_uniform":
        return advantage
    elif weight_type == "grpo_guard":
        return 0
    elif weight_type == "a_t_exp_a":
        return advantage * (t ** config.off_policy_weight_exp)
    elif weight_type == "1_t_exp_a":
        return t ** config.off_policy_weight_exp
    elif weight_type == "a_1t_exp_a":
        return advantage * ((1.0 - t) ** config.off_policy_weight_exp)
    elif weight_type == "1_1t_exp_a":
        return (1.0 - t) ** config.off_policy_weight_exp
    elif weight_type == "c_1t_exp_a":
        return config.off_policy_const_factor * (1.0 - t) ** config.off_policy_weight_exp
    elif weight_type == "None" or weight_type is None:
        return 0.0
    else:
        raise NotImplementedError(f"Unknown weight type: {weight_type}")


def get_lr_factor(global_step, config):
    scheduler_type = config.train.lr_scheduler
    warmup_steps = config.train.lr_warmup_steps
    step = global_step

    if step < warmup_steps:
        return step / max(1, warmup_steps)

    if scheduler_type == "constant":
        return 1.0
    elif scheduler_type == "cosine":
        total_steps = config.train.lr_total_steps
        assert total_steps > 0, "lr_total_steps must be > 0 for cosine scheduler"
        min_lr_ratio = config.train.lr_min_ratio
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    elif scheduler_type == "linear":
        total_steps = config.train.lr_total_steps
        min_lr_ratio = config.train.lr_min_ratio
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
    elif scheduler_type == "step_halving":
        halving_steps = list(config.train.lr_halving_steps)
        n_halvings = sum(1 for s in halving_steps if step >= s)
        return 0.5 ** n_halvings
    else:
        raise ValueError(f"Unknown lr_scheduler type: {scheduler_type}")


def set_lr(optimizer, config, global_step):
    factor = get_lr_factor(global_step, config)
    lr = config.train.learning_rate * factor
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f.readlines()]
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
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
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
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards["ori_avg"][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], "big")
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def compute_log_prob(transformer, pipeline, sample, j, config, rank):
    img_shapes = [
        [(1, config.resolution // pipeline.vae_scale_factor // 2, config.resolution // pipeline.vae_scale_factor // 2)]
    ] * len(sample["latents"][:, j])
    do_true_cfg = (
        config.train.cfg and config.sample.guidance_scale > 1.0
    )
    txt_seq_lens = sample["prompt_embeds_mask"].sum(dim=1).tolist()
    if do_true_cfg:
        negative_txt_seq_lens = (
            sample["negative_prompt_embeds_mask"].sum(dim=1).tolist()
        )
        max_seq_len = max(txt_seq_lens + negative_txt_seq_lens)
    else:
        max_seq_len = max(txt_seq_lens)

    prompt_embeds_mask = sample["prompt_embeds_mask"][:, :max_seq_len]
    prompt_embeds = sample["prompt_embeds"][:, :max_seq_len]
    latents = sample["latents"][:, j]
    timesteps = sample["timesteps"][:, j]
    if do_true_cfg:
        negative_prompt_embeds_mask = sample[
            "negative_prompt_embeds_mask"
        ][:, :max_seq_len]
        negative_prompt_embeds = sample["negative_prompt_embeds"][
            :, :max_seq_len
        ]
        model_latents = torch.cat([latents, latents], dim=0)
        model_timesteps = torch.cat([timesteps, timesteps], dim=0)
        model_prompt_embeds_mask = torch.cat(
            [prompt_embeds_mask, negative_prompt_embeds_mask], dim=0
        )
        model_prompt_embeds = torch.cat(
            [prompt_embeds, negative_prompt_embeds], dim=0
        )
        model_img_shapes = img_shapes * 2
    else:
        model_latents = latents
        model_timesteps = timesteps
        model_prompt_embeds_mask = prompt_embeds_mask
        model_prompt_embeds = prompt_embeds
        model_img_shapes = img_shapes

    noise_pred = transformer(
        hidden_states=model_latents,
        timestep=model_timesteps / 1000,
        guidance=None,
        encoder_hidden_states_mask=model_prompt_embeds_mask,
        encoder_hidden_states=model_prompt_embeds,
        img_shapes=model_img_shapes,
        attention_kwargs=pipeline.attention_kwargs,
        return_dict=False,
    )[0]
    if do_true_cfg:
        noise_pred, neg_noise_pred = noise_pred.chunk(2, dim=0)
        comb_pred = neg_noise_pred + config.sample.guidance_scale * (
            noise_pred - neg_noise_pred
        )
        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        velocity = comb_pred * (cond_norm / noise_norm)
    else:
        velocity = noise_pred

    prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = sde_step_with_logprob(
        pipeline.scheduler,
        velocity.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
        return_sqrt_dt=True,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t, velocity, sqrt_dt


def eval(pipeline, test_dataloader, config, rank, local_rank, world_size, device, global_step, reward_fn, executor, autocast, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    all_rewards = defaultdict(list)
    for test_batch in tqdm(
        test_dataloader,
        desc="Eval: ",
        disable=local_rank != 0,
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        with autocast():
            with torch.no_grad():
                collected_data = pipeline_with_logprob(
                    pipeline,
                    prompts,
                    negative_prompt=[" "] * len(prompts),
                    num_inference_steps=config.sample.eval_num_steps,
                    true_cfg_scale=config.sample.guidance_scale,
                    cfg_enabled=config.train.cfg,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                    sde_window_size=0,
                )
        images = collected_data["images"]
        if getattr(reward_fn, "uses_training_device", False):
            rewards = ImmediateResult(reward_fn(images, prompts, prompt_metadata, only_strict=False))
        else:
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
            time.sleep(0)
        rewards, reward_metadata = rewards.result()
        offload_reward_fn = getattr(reward_fn, "offload", None)
        if offload_reward_fn is not None:
            offload_reward_fn()

        for key, value in rewards.items():
            rewards_gather = gather_tensor(torch.as_tensor(value, device=device), world_size).cpu().float().numpy()
            all_rewards[key].append(rewards_gather)

    last_batch_images_gather = gather_tensor(torch.as_tensor(images, device=device), world_size).cpu().float().numpy()
    last_batch_prompts_gather = gather_prompt_strings(prompts, world_size)
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = gather_tensor(torch.as_tensor(value, device=device), world_size).cpu().float().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if rank == 0:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(EVAL_LOG_IMAGE_COUNT, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            wandb.log(
                {
                    "eval/images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(
                                f"{k}: {v:.2f}"
                                for k, v in reward.items()
                                if v != -10
                            ),
                        )
                        for idx, (prompt, reward) in enumerate(
                            zip(sampled_prompts, sampled_rewards)
                        )
                    ],
                    **{
                        f"eval/reward_{key}": np.mean(value[value != -10])
                        for key, value in all_rewards.items()
                    },
                },
                step=global_step,
                commit=False,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)


def get_transformer_layer_cls():
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionBlock, Qwen2_5_VLDecoderLayer
    return {
        QwenImageTransformerBlock,
        Qwen2_5_VLVisionBlock,
        Qwen2_5_VLDecoderLayer,
    }


QWEN_TRAINER_NAME = "train_qwenimage_kde"
RESUME_MODEL_NAME = "model.safetensors"
RESUME_OPTIMIZER_NAME = "resume_optimizer.pt"
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


def save_safetensors_staged(state_dict, destination_path):
    file_descriptor, local_path = tempfile.mkstemp(
        prefix="qwenimage_checkpoint_",
        suffix=".safetensors",
        dir=checkpoint_local_tmp_dir(),
    )
    os.close(file_descriptor)
    try:
        save_file(state_dict, local_path)
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
        prefix="qwenimage_resume_",
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


def resolve_resume_checkpoint(config, rank, world_size):
    expected = expected_metadata(
        config,
        trainer=QWEN_TRAINER_NAME,
        world_size=world_size,
    )
    search_root = config.resume_from if config.resume_from else config.save_dir
    checkpoint_holder = [None]
    if rank == 0:
        checkpoint_holder[0] = find_latest_checkpoint(
            search_root,
            expected,
            include_checkpoints_subdir=False,
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


def load_resume_checkpoint(
    checkpoint_dir,
    optimizer,
    optimizer_offload_hook,
    config,
    train_sampler,
    rank,
    world_size,
):
    metadata = load_metadata(checkpoint_dir)
    expected = expected_metadata(
        config,
        trainer=QWEN_TRAINER_NAME,
        world_size=world_size,
    )
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Checkpoint metadata mismatch for {key}: "
                f"checkpoint={metadata.get(key)!r}, current={value!r}"
            )

    optimizer_state_path = os.path.join(
        checkpoint_dir,
        f"{RESUME_OPTIMIZER_NAME}.{rank}",
    )
    optimizer.load_state_dict(
        torch_load(optimizer_state_path, map_location="cpu")
    )
    if optimizer_offload_hook is not None:
        optimizer_offload_hook.post_step_hook(optimizer, (), {})

    rank_state_path = os.path.join(
        checkpoint_dir,
        f"{RESUME_TRAINING_STATE_NAME}.{rank}",
    )
    rank_state = torch_load(rank_state_path, map_location="cpu")
    if rank_state["rank"] != rank:
        raise ValueError(f"Rank state mismatch in {rank_state_path}")
    train_sampler.set_epoch(rank_state["sampler_epoch"])
    load_rng_state_dict(rank_state["rng_state"])
    dist.barrier()
    return rank_state["epoch"], rank_state["global_step"]


def save_training_checkpoint(
    save_dir,
    transformer,
    optimizer,
    optimizer_offload_hook,
    config,
    train_sampler,
    epoch,
    global_step,
    rank,
    world_size,
):
    save_root = os.path.join(save_dir, f"checkpoint-{global_step}")
    tmp_root = os.path.join(save_dir, f".checkpoint-{global_step}.tmp")
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
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root)
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()
    os.makedirs(tmp_root, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with FSDP.state_dict_type(
        transformer,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        model_state = transformer.state_dict()
        if rank == 0:
            save_safetensors_staged(
                model_state,
                os.path.join(tmp_root, RESUME_MODEL_NAME),
            )
        del model_state

    torch.save(
        optimizer_state_for_save(optimizer, optimizer_offload_hook),
        os.path.join(tmp_root, f"{RESUME_OPTIMIZER_NAME}.{rank}"),
    )
    rank_state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "rank": int(rank),
        "rng_state": rng_state_dict(),
        "sampler_epoch": int(train_sampler.epoch),
    }
    torch.save(
        rank_state,
        os.path.join(tmp_root, f"{RESUME_TRAINING_STATE_NAME}.{rank}"),
    )
    dist.barrier()

    if rank == 0:
        metadata = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "sampler_epoch": int(train_sampler.epoch),
            **expected_metadata(
                config,
                trainer=QWEN_TRAINER_NAME,
                world_size=world_size,
            ),
        }
        with open(
            os.path.join(tmp_root, META_NAME),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
        with open(
            os.path.join(tmp_root, COMPLETE_NAME),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {"epoch": int(epoch), "global_step": int(global_step)},
                file,
                sort_keys=True,
            )
        os.replace(tmp_root, save_root)
    dist.barrier()
    return save_root


def main(_):
    config = FLAGS.config

    is_distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    resume_checkpoint = resolve_resume_checkpoint(config, rank, world_size)

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # Number of timesteps to train on per trajectory
    if config.sample.sde_window_size > 0:
        num_train_timesteps = config.sample.sde_window_size
    else:
        num_train_timesteps = config.sample.num_steps - 1

    project_dir = os.path.join(config.logdir, config.run_name)
    os.makedirs(project_dir, exist_ok=True)
    if rank == 0:
        wandb.init(
            project="diffusion-rl",
            name=config.run_name,
            config=config.to_dict(),
        )
        logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    inference_dtype = torch.float32
    if config.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline = DiffusionPipeline.from_pretrained(config.pretrained.model, torch_dtype=inference_dtype)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=local_rank != 0,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    pipeline.vae.to(device, dtype=torch.float32)

    # Deep-copy the base transformer BEFORE LoRA is applied so transformer_ref
    # holds the original frozen weights without any network access.
    transformer_ref_base = None
    if config.train.beta > 0:
        transformer_ref_base = copy.deepcopy(pipeline.transformer)
        transformer_ref_base.eval()
        transformer_ref_base.requires_grad_(False)

    if config.use_lora:
        target_modules = [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "img_mlp.net.0.proj", "img_mlp.net.2",
            "txt_mlp.net.0.proj", "txt_mlp.net.2",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

    transformer = pipeline.transformer
    if resume_checkpoint is not None:
        if rank == 0:
            with load_safetensors_staged(
                os.path.join(resume_checkpoint, RESUME_MODEL_NAME)
            ) as model_state:
                transformer.load_state_dict(
                    model_state
                )
        dist.barrier()

    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        cpu_offload=False,
        num_replicate=1,
        num_shard=world_size,
        mixed_precision_dtype=inference_dtype,
        use_activation_checkpointing=config.activation_checkpointing,
        use_device_mesh=False,
        sync_module_states=resume_checkpoint is not None,
    )
    transformer.cpu().to(dtype=torch.float32)
    transformer = fsdp_wrapper(transformer, fsdp_config, get_transformer_layer_cls)
    pipeline.transformer = transformer

    transformer_ref = None
    if config.train.beta > 0:
        transformer_ref = transformer_ref_base
        transformer_ref.cpu().to(dtype=torch.float32)
        transformer_ref = fsdp_wrapper(transformer_ref, fsdp_config, get_transformer_layer_cls)

    pipeline.text_encoder.cpu().to(dtype=torch.float32)
    pipeline.text_encoder = fsdp_wrapper(pipeline.text_encoder, fsdp_config, get_transformer_layer_cls)

    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    ema = None

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
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
    optimizer_offload_handles = []
    if config.fsdp_optimizer_offload:
        optimizer_offload_handles, optimizer_offload_hook = (
            register_optimizer_offload_hooks(optimizer)
        )

    train_dataloader_generator = torch.Generator().manual_seed(
        config.seed + rank
    )
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=world_size,
            rank=rank,
            seed=config.seed,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
            collate_fn=TextPromptDataset.collate_fn,
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
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=world_size,
            rank=rank,
            seed=config.seed,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
            collate_fn=GenevalPromptDataset.collate_fn,
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
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr and geneval are supported with dataset")

    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    if config.mixed_precision == "fp16":
        autocast = lambda: torch.cuda.amp.autocast(dtype=torch.float16)
    elif config.mixed_precision == "bf16":
        autocast = lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(
        device,
        config.reward_fn,
        config.reward_backends,
        hpsv3_offload=config.hpsv3_offload,
    )
    if "hpsv3" in config.reward_fn and not config.hpsv3_offload:
        eval_reward_fn = reward_fn
    else:
        eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(
            device,
            config.reward_fn,
            config.reward_backends,
            hpsv3_offload=config.hpsv3_offload,
        )

    executor = futures.ThreadPoolExecutor(max_workers=8)

    samples_per_epoch = (
        config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size * world_size * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)
    if resume_checkpoint is not None:
        logger.info(f"Resuming training from {resume_checkpoint}")
        epoch, global_step = load_resume_checkpoint(
            resume_checkpoint,
            optimizer,
            optimizer_offload_hook,
            config,
            train_sampler,
            rank,
            world_size,
        )
    set_lr(optimizer, config, global_step)
    last_checkpoint_step = global_step if resume_checkpoint is not None else -1

    while epoch < config.num_epochs:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if config.eval and epoch % config.eval_freq == 0:
            eval(
                pipeline, test_dataloader, config, rank, local_rank, world_size,
                device, global_step, eval_reward_fn, executor, autocast,
                ema, transformer_trainable_parameters,
            )

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        epoch_prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=local_rank != 0,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)
            epoch_prompts.extend(prompts)

            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch * 10000 + i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    collected_data = pipeline_with_logprob(
                        pipeline,
                        prompts,
                        negative_prompt=[" "] * len(prompts),
                        num_inference_steps=config.sample.num_steps,
                        true_cfg_scale=config.sample.guidance_scale,
                        cfg_enabled=config.train.cfg,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator,
                        sde_window_size=config.sample.sde_window_size,
                        sde_window_range=config.sample.sde_window_range,
                        return_velocity=True,
                    )

            latents = torch.stack(collected_data["all_latents"], dim=1)
            log_probs = torch.stack(collected_data["all_log_probs"], dim=1)
            timesteps = torch.stack(collected_data["all_timesteps"]).unsqueeze(0).repeat(
                config.sample.train_batch_size, 1
            )
            velocities = torch.stack(collected_data["all_velocity"], dim=1)
            images = collected_data["images"]

            if getattr(reward_fn, "uses_training_device", False):
                rewards = ImmediateResult(reward_fn(images, prompts, prompt_metadata, only_strict=True))
            else:
                rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
                time.sleep(0)

            samples.append(
                {
                    "prompt_embeds": collected_data["prompt_embeds"],
                    "prompt_embeds_mask": collected_data["prompt_embeds_mask"],
                    "negative_prompt_embeds": collected_data["negative_prompt_embeds"],
                    "negative_prompt_embeds_mask": collected_data["negative_prompt_embeds_mask"],
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "velocities": velocities,
                    "clean_latents": latents[:, -1],
                    "rewards": rewards,
                }
            )

        max_prompt_embeds_len = max([sample["prompt_embeds_mask"].shape[1] for sample in samples])

        for sample in tqdm(samples, desc="Waiting for rewards", disable=local_rank != 0, position=0):
            seq_pad_len = max_prompt_embeds_len - sample["prompt_embeds"].shape[1]
            sample["prompt_embeds"] = torch.nn.functional.pad(
                sample["prompt_embeds"], (0, 0, 0, seq_pad_len), value=0
            )
            sample["prompt_embeds_mask"] = torch.nn.functional.pad(
                sample["prompt_embeds_mask"], (0, seq_pad_len), value=0
            )
            if config.train.cfg and config.sample.guidance_scale > 1.0:
                sample["negative_prompt_embeds"] = torch.nn.functional.pad(
                    sample["negative_prompt_embeds"],
                    (0, 0, 0, seq_pad_len),
                    value=0,
                )
                sample["negative_prompt_embeds_mask"] = (
                    torch.nn.functional.pad(
                        sample["negative_prompt_embeds_mask"],
                        (0, seq_pad_len),
                        value=0,
                    )
                )
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=device).float()
                for key, value in rewards.items()
            }
        offload_reward_fn = getattr(reward_fn, "offload", None)
        if offload_reward_fn is not None:
            offload_reward_fn()

        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and rank == 0:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_imgs = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_imgs)
                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                sampled_prompts = [epoch_prompts[i] for i in sample_indices]
                sampled_rewards = [rewards["avg"][i] for i in sample_indices]
                wandb.log(
                    {
                        "train/images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt[:100]} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(
                                zip(sampled_prompts, sampled_rewards)
                            )
                        ],
                    },
                    step=global_step,
                    commit=False,
                )

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        gathered_rewards = {key: gather_tensor(value, world_size) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().float().numpy() for key, value in gathered_rewards.items()}
        if rank == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward/{key}": value.mean()
                        for key, value in gathered_rewards.items()
                        if "_strict_accuracy" not in key and "_accuracy" not in key
                    },
                },
                step=global_step,
                commit=False,
            )

        # Preserve the full original strings so long prompts cannot collide after
        # tokenization, truncation, and decoding.
        prompts = gather_prompt_strings(epoch_prompts, world_size)

        if config.per_prompt_stat_tracking:
            advantages = stat_tracker.update(prompts, gathered_rewards["avg"])
            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)
            if rank == 0:
                wandb.log(
                    {
                        "stats/group_size": group_size,
                        "stats/trained_prompt_num": trained_prompt_num,
                        "stats/zero_std_ratio": zero_std_ratio,
                        "stats/reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                    commit=False,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards["avg"] - gathered_rewards["avg"].mean()) / (
                gathered_rewards["avg"].std() + 1e-4
            )

        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(world_size, -1, advantages.shape[-1])[rank].to(device)
        )

        del samples["rewards"]

        # Determine which timesteps use the KDE value-gradient estimator
        gradient_train_timesteps = list(range(config.start_timesteps, config.end_timesteps))
        kde_timesteps = [
            step_index
            for step_index in gradient_train_timesteps
            if get_timestep_config_value(config.value_gradient_estimator, step_index) == "kde"
        ]
        use_kde = len(kde_timesteps) > 0

        if use_kde:
            with torch.no_grad():
                prompt_to_group = {prompt: gid for gid, prompt in enumerate(dict.fromkeys(prompts))}
                prompt_group_ids = torch.as_tensor(
                    [prompt_to_group[p] for p in prompts], device=device
                )
                local_prompt_group_ids = prompt_group_ids.reshape(world_size, -1)[rank]
                kde_estimator, kde_stats = compute_kde_estimator(
                    local_prompt_group_ids=local_prompt_group_ids,
                    local_latents=samples["latents"],
                    local_velocities=samples["velocities"],
                    local_timesteps=samples["timesteps"],
                    global_prompt_group_ids=prompt_group_ids,
                    global_clean_latents=gather_tensor(samples["clean_latents"], world_size),
                    global_advantages=gather_tensor(samples["advantages"], world_size),
                    adv_clip_max=config.train.adv_clip_max,
                    step_indices=kde_timesteps,
                    kde_bandwidth=config.kde_bandwidth,
                )
                samples["kde_estimator"] = kde_estimator
                gathered_kde_stats = {
                    stat_name: gather_tensor(stat_values, world_size)
                    for stat_name, stat_values in kde_stats.items()
                }
                kde_stat_tags = {
                    "weight_entropy": "train/kde_weight_entropy",
                    "logits_mean": "train/kde_logits_mean",
                    "logits_max": "train/kde_logits_max",
                    "logits_min": "train/kde_logits_min",
                }
                kde_log = {}
                for stat_name, stat_values in gathered_kde_stats.items():
                    base_tag = kde_stat_tags[stat_name]
                    overall_mean = stat_values.mean()
                    if rank == 0:
                        kde_log[base_tag] = overall_mean.item()
                    for stats_index, step_index in enumerate(kde_timesteps):
                        timestep_tag = f"{base_tag}/timestep_{step_index}"
                        timestep_mean = stat_values[:, stats_index].mean()
                        if rank == 0:
                            kde_log[timestep_tag] = timestep_mean.item()
                if rank == 0:
                    wandb.log(kde_log, step=global_step, commit=False)
            if rank == 0:
                wandb.log(
                    {
                        "train/actual_batch_size": samples["advantages"].shape[0]
                        // config.sample.num_batches_per_epoch
                    },
                    step=global_step,
                    commit=False,
                )
        else:
            num_batches = config.sample.num_batches_per_epoch
            mask = samples["advantages"].abs().sum(dim=1) != 0
            true_count = mask.sum()
            if true_count % num_batches != 0:
                false_indices = torch.where(~mask)[0]
                num_to_change = num_batches - (true_count % num_batches)
                if len(false_indices) >= num_to_change:
                    random_indices = torch.randperm(len(false_indices))[:num_to_change]
                    mask[false_indices[random_indices]] = True
            if rank == 0:
                wandb.log(
                    {"train/actual_batch_size": mask.sum().item() // num_batches},
                    step=global_step,
                    commit=False,
                )
            samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        # Each optimizer step covers gradient_accumulation_steps forward passes
        # (config.train.gradient_accumulation_steps mini-batches × len(gradient_train_timesteps) timesteps)
        gradient_accumulation_steps = config.train.gradient_accumulation_steps * len(gradient_train_timesteps)

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            num_training_batches = config.sample.num_batches_per_epoch
            training_batch_size = total_batch_size // num_training_batches
            samples_batched = {
                k: v.reshape(
                    num_training_batches,
                    training_batch_size,
                    *v.shape[1:],
                )
                for k, v in samples.items()
            }
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=local_rank != 0,
            ):
                for step_num, j in enumerate(tqdm(
                    gradient_train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=local_rank != 0,
                )):
                    should_sync = (i * len(gradient_train_timesteps) + step_num + 1) % gradient_accumulation_steps == 0

                    with autocast():
                        prev_sample, log_prob, prev_sample_mean, std_dev_t, velocity, sqrt_dt = compute_log_prob(
                            transformer, pipeline, sample, j, config, rank
                        )
                        velocity_ref = None
                        if config.train.beta > 0:
                            with torch.no_grad():
                                _, _, prev_sample_mean_ref, _, velocity_ref, _ = compute_log_prob(
                                    transformer_ref, pipeline, sample, j, config, rank
                                )

                    advantages = torch.clamp(
                        sample["advantages"][:, j],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    t_j = sample["timesteps"][:, j].mean() / 1000.0
                    t_j_max = sample["timesteps"][:, 1].mean() / 1000.0
                    # spatial_dims covers all non-batch dimensions: (1, 2) for QwenImage (seq, dim)
                    spatial_dims = tuple(range(1, velocity.ndim))

                    value_grad_estimator_type = get_timestep_config_value(config.value_gradient_estimator, j)
                    if value_grad_estimator_type == "kde":
                        value_grad_estimator = sample["kde_estimator"][:, j]
                    else:
                        raise NotImplementedError(f"Unknown value-gradient estimator: {value_grad_estimator_type}")

                    off_policy_weight_type = get_timestep_config_value(config.off_policy_weight, j)
                    off_policy_weight = get_off_policy_weight(
                        t_j, advantages, config, off_policy_weight_type,
                        t_j_max=t_j_max, sqrt_dt=sqrt_dt.mean(),
                    )
                    policy_gradient_weight_type = get_timestep_config_value(config.policy_gradient_weight, j)
                    policy_gradient_weight = get_policy_gradient_weight(
                        t_j, config, policy_gradient_weight_type,
                        t_j_max=t_j_max, sqrt_dt=sqrt_dt.mean(),
                    )

                    v_diff = velocity - sample["velocities"][:, j]
                    off_policy_loss = off_policy_weight * (v_diff**2).mean(dim=spatial_dims)
                    policy_gradient_loss = policy_gradient_weight * (v_diff * value_grad_estimator).mean(dim=spatial_dims)
                    policy_loss = torch.mean(policy_gradient_loss + off_policy_loss) / gradient_accumulation_steps

                    if config.train.beta > 0:
                        if config.reweight_kl:
                            kl_loss = ((velocity - velocity_ref)**2).mean(dim=spatial_dims, keepdim=True)
                        else:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref)**2).mean(dim=spatial_dims, keepdim=True) / (2 * std_dev_t**2)
                        loss = policy_loss + config.train.beta * torch.mean(kl_loss) / gradient_accumulation_steps
                    else:
                        loss = policy_loss

                    info["approx_kl"].append(
                        0.5 * torch.mean((log_prob - sample["log_probs"][:, j])**2)
                    )
                    info["policy_loss"].append(policy_loss)
                    if config.train.beta > 0:
                        info["kl_loss"].append(config.train.beta * torch.mean(kl_loss) / gradient_accumulation_steps)
                    info["loss"].append(loss)

                    loss.backward()
                    if should_sync:
                        torch.nn.utils.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        if is_distributed:
                            for k, v in info.items():
                                dist.all_reduce(v, op=dist.ReduceOp.SUM)
                                info[k] = v / world_size
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if rank == 0:
                            wandb.log(
                                {
                                    **{
                                        f"train/{k}": (
                                            v.item()
                                            if isinstance(v, torch.Tensor)
                                            else v
                                        )
                                        for k, v in info.items()
                                    },
                                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                                },
                                step=global_step,
                                commit=True,
                            )
                        global_step += 1
                        set_lr(optimizer, config, global_step)
                        info = defaultdict(list)

                if config.train.ema and ema is not None:
                    ema.step(transformer_trainable_parameters, global_step)

        epoch += 1
        if (
            config.save
            and epoch % config.save_freq == 0
            and global_step > last_checkpoint_step
        ):
            if not should_sync:
                raise RuntimeError(
                    "Cannot save a resumable checkpoint with pending accumulated gradients"
                )
            save_training_checkpoint(
                config.save_dir,
                transformer,
                optimizer,
                optimizer_offload_hook,
                config,
                train_sampler,
                epoch,
                global_step,
                rank,
                world_size,
            )
            last_checkpoint_step = global_step

    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)

