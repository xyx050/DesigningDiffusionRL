# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NFT (Negative Flow Training) training script for Stable Diffusion 3.
# Ported from DiffusionNFT/scripts/train_nft_sd3.py to use Accelerate for distributed training,
# matching the package conventions of flow_grpo/scripts/train_sd3.py.
#
# Key algorithmic differences from train_sd3.py (GRPO):
#   - Loss: x0-prediction flow matching loss (not PPO-style log-prob ratio loss)
#   - Model: Dual LoRA adapters ("default" = current policy, "old" = decaying reference)
#   - Sampling: Uses "old" adapter with deterministic dpm2 solver (guidance_scale=1.0)
#   - Decay: After each epoch, old = old * decay + default * (1 - decay)

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.nft_sd3_pipeline_with_logprob import pipeline_with_logprob as nft_pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
from flow_grpo.logging_utils import ExperimentLogger

wandb = ExperimentLogger()
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
import os
import shutil
import sys
import shlex

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def _apply_fix_config(config):
    if "fix" not in config or not config.fix:
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

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
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]

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
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
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


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def return_decay(step, decay_type):
    """Compute old-adapter decay coefficient based on global step and decay schedule type."""
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
        assert False, f"Unknown decay_type: {decay_type}"

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    gathered_rewards["avg"] has shape (N, T); uses the first timestep for std computation.
    """
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    # Use first timestep rewards for std calculation
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def eval(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    accelerator,
    global_step,
    reward_fn,
    executor,
    autocast,
    ema,
    transformer_trainable_parameters,
):
    """Run evaluation: sample with default adapter (deterministic, flow solver) and log rewards."""
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    # Make sure we're using the default (current policy) adapter for eval
    pipeline.transformer.set_adapter("default")
    pipeline.transformer.eval()

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)
    for test_batch in tqdm(
        test_dataloader,
        desc="Eval: ",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
        )
        # Handle last batch (may be smaller)
        cur_bs = len(prompt_embeds)
        cur_neg_embeds = sample_neg_prompt_embeds[:cur_bs]
        cur_neg_pooled = sample_neg_pooled_prompt_embeds[:cur_bs]

        with autocast():
            with torch.no_grad():
                images, _, _ = nft_pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=cur_neg_embeds,
                    negative_pooled_prompt_embeds=cur_neg_pooled,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                )

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(
                torch.as_tensor(value, device=accelerator.device)
            ).cpu().numpy()
            all_rewards[key].append(rewards_gather)

    last_batch_images_gather = accelerator.gather(
        torch.as_tensor(images, device=accelerator.device)
    ).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(
            torch.as_tensor(value, device=accelerator.device)
        ).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}

    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [
                {k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather}
                for index in sample_indices
            ]
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

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)


def main(_):
    config = FLAGS.config
    _apply_fix_config(config)
    if config.loss_type not in ("nft", "awm"):
        raise ValueError(f"Unknown loss_type: {config.loss_type}")
    if config.train.ema_beta > 0 and config.train.kl_ema_weight != "Uniform":
        raise ValueError(f"Unknown kl_ema_weight: {config.train.kl_ema_weight}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
    if config.loss_type != "nft":
        config.run_name += f"_{config.loss_type}"

    # Number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    # Each gradient accumulation cycle covers all timesteps × sample-level accumulation steps
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )

    if accelerator.is_main_process:
        wandb.init(
            project="diffusion-rl",
            name=config.run_name,
            config=config.to_dict(),
        )

        code_save_dir = os.path.join(config.logdir, config.run_name, "saved_setting")
        os.makedirs(code_save_dir, exist_ok=True)

        script_path = os.path.abspath(__file__)
        shutil.copy(script_path, os.path.join(code_save_dir, os.path.basename(script_path)))

        with open(os.path.join(code_save_dir, "argv.txt"), "w") as f:
            f.write(" ".join(shlex.quote(a) for a in sys.argv) + "\n")

        d = FLAGS.flag_values_dict()
        try:
            d["config"] = d["config"].to_dict()
        except Exception:
            d["config"] = str(d["config"])
        with open(os.path.join(code_save_dir, "flags.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, sort_keys=True, ensure_ascii=False)

        logger.info(f"Saved training script and config to {code_save_dir}")
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    # Load pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)

    # Freeze non-trainable components
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

        # Add a second "old" adapter that will serve as the decaying reference model
        pipeline.transformer.add_adapter("old", transformer_lora_config)
        if config.train.ema_beta > 0:
            pipeline.transformer.add_adapter("ema", transformer_lora_config)
        pipeline.transformer.set_adapter("default")
    else:
        assert False, "NFT requires LoRA (use_lora=True) for the dual-adapter mechanism."

    transformer = pipeline.transformer

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Collect trainable parameters per adapter (before accelerator.prepare changes the wrapper)
    transformer.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    ema_transformer_trainable_parameters = None
    if config.train.ema_beta > 0:
        transformer.set_adapter("ema")
        ema_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer.set_adapter("default")

    # EMA tracks the "default" adapter parameters
    ema = EMAModuleWrapper(
        transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=accelerator.device
    ) if config.train.ema else None

    # Optimizer only updates "default" adapter
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

    # Reward functions
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(
        accelerator.device, config.reward_fn, config.reward_backends
    )
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(
        accelerator.device, config.reward_fn, config.reward_backends
    )

    # Datasets and dataloaders
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')
        collate_fn = TextPromptDataset.collate_fn
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')
        collate_fn = GenevalPromptDataset.collate_fn
    else:
        raise NotImplementedError(f"prompt_fn '{config.prompt_fn}' is not supported with dataset.")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=config.seed,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=1,
    )

    # Negative prompt embeddings (for CFG; with guidance_scale=1.0 they are not used but kept for API compat)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    assert config.per_prompt_stat_tracking, "NFT requires per_prompt_stat_tracking=True."
    stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # autocast: for non-lora training must use accelerator.autocast; for lora can use nullcontext
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare with accelerator (handles DDP wrapping, device placement, etc.).
    # NOTE: train_dataloader is intentionally NOT prepared. accelerate wraps a
    # prepared loader in DataLoaderShard, whose __iter__ reads one batch ahead;
    # combined with our per-batch train_sampler.set_epoch(), that prefetch would
    # make set_epoch lag by one step (i>=1), desyncing the sampled prompt groups
    # from the official script. The official script uses a bare DataLoader and the
    # custom DistributedKRepeatSampler already shards by rank, so keeping it bare
    # (with num_workers=0) reproduces the official sampling order exactly. The
    # train batch is just text prompts (no tensors), so no device placement is
    # needed; the infinite sampler never hits end_of_dataloader, so gradient
    # accumulation timing (driven by the 144-step counter) is unaffected.
    transformer, optimizer, test_dataloader = accelerator.prepare(
        transformer, optimizer, test_dataloader
    )

    executor = futures.ThreadPoolExecutor(max_workers=2)

    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running SD3 NFT/AWM training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (parallel + accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    logger.info(f"  loss_type: {config.loss_type}")
    logger.info(f"  train.ema_beta: {config.train.ema_beta}")
    logger.info(f"  Solver: {config.sample.solver}, deterministic: {config.sample.deterministic}")
    logger.info(f"  decay_type: {config.decay_type}, beta (blend): {config.beta}")
    logger.info(f"  train.adv_mode: {config.train.adv_mode}")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    # Initialize "old" adapter weights from "default" adapter
    with torch.no_grad():
        for src_param, tgt_param in zip(
            transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
        ):
            tgt_param.data.copy_(src_param.detach().data)
            assert src_param is not tgt_param
        if config.train.ema_beta > 0:
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, ema_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(src_param.detach().data)
                assert src_param is not tgt_param

    while epoch < config.num_epochs:
        #################### EVAL ####################
        pipeline.transformer.set_adapter("default")
        pipeline.transformer.eval()
        if config.eval and epoch % config.eval_freq == 0:
            eval(
                pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator,
                global_step, eval_reward_fn, executor, autocast, ema, transformer_trainable_parameters,
            )
        if config.save and epoch % config.save_freq == 0 and epoch > 0:
            save_ckpt(
                config.save_dir, transformer, global_step, accelerator, ema,
                transformer_trainable_parameters, config,
            )

        #################### SAMPLING (with "old" adapter) ####################
        pipeline.transformer.set_adapter("old")
        pipeline.transformer.eval()
        samples = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt",
            ).input_ids.to(accelerator.device)

            with autocast():
                with torch.no_grad():
                    images, latents, _ = nft_pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[:len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[:len(prompts)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=config.sample.deterministic,
                        solver=config.sample.solver,
                        model_type="sd3",
                    )

            # latents is a list of (batch, C, H, W) tensors; stack to (batch, num_steps+1, C, H, W)
            latents_stacked = torch.stack(latents, dim=1)
            latents_clean = latents_stacked[:, -1]  # x0: final denoised latent

            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(accelerator.device)
            next_timesteps = torch.cat(
                [timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1
            )

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "next_timesteps": next_timesteps,
                    "latents_clean": latents_clean,
                    "rewards": rewards_future,
                }
            )

        # Wait for all rewards
        for sample in tqdm(
            samples, desc="Waiting for rewards", disable=not accelerator.is_local_main_process, position=0
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # Collate samples into a single dict
        collated_samples = {
            k: (
                torch.cat([s[k] for s in samples], dim=0)
                if not isinstance(samples[0][k], dict)
                else {sk: torch.cat([s[k][sk] for s in samples], dim=0) for sk in samples[0][k]}
            )
            for k in samples[0].keys()
        }

        # Log sample images
        if epoch % 10 == 0 and accelerator.is_main_process:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images))
                for idx in range(num_to_log):
                    img_data = images[idx].cpu()
                    pil = Image.fromarray((img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                wandb.log(
                    {
                        "train/images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=(
                                    f"{prompts[idx]:.100} | avg: "
                                    f"{collated_samples['rewards']['avg'][-len(images) + idx]:.2f}"
                                ),
                            )
                            for idx in range(num_to_log)
                        ],
                    },
                    step=global_step,
                    commit=False,
                )

        # Expand avg reward along timestep dimension for per-timestep advantage tracking
        collated_samples["rewards"]["ori_avg"] = collated_samples["rewards"]["avg"].clone()
        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )

        # Gather rewards across processes
        gathered_rewards = {
            key: accelerator.gather(value).cpu().numpy()
            for key, value in collated_samples["rewards"].items()
        }

        if accelerator.is_main_process:
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

        # Per-prompt advantage computation
        prompt_ids_gathered = accelerator.gather(collated_samples["prompt_ids"]).cpu().numpy()
        prompts_all = pipeline.tokenizer.batch_decode(prompt_ids_gathered, skip_special_tokens=True)

        advantages = stat_tracker.update(prompts_all, gathered_rewards["avg"])

        group_size, trained_prompt_num = stat_tracker.get_stats()
        zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all, gathered_rewards)

        if accelerator.is_main_process:
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

        # Distribute advantages back to this process
        advantages = torch.as_tensor(advantages)
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        assert advantages.shape[0] == accelerator.num_processes * samples_per_gpu
        collated_samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, samples_per_gpu, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        total_batch_size_local, num_timesteps = collated_samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        # Re-enable default adapter for training
        pipeline.transformer.set_adapter("default")
        transformer.train()

        for inner_epoch in range(config.train.num_inner_epochs):
            # Shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size_local, device=accelerator.device)
            shuffled = {k: v[perm] for k, v in collated_samples.items()}

            # Also shuffle timesteps independently per sample (NFT-style).
            # Permute the FULL num_timesteps (=num_steps) columns, then the timestep
            # loop below consumes only the first num_train_timesteps of them. This
            # makes each sample drop a RANDOM 1-of-num_steps timestep per epoch
            # (matching the official script), instead of always dropping the last one.
            perms_time = torch.stack(
                [torch.randperm(num_timesteps, device=accelerator.device)
                 for _ in range(total_batch_size_local)]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled[key] = shuffled[key][
                    torch.arange(total_batch_size_local, device=accelerator.device)[:, None], perms_time
                ]

            # Rebatch for training
            num_batches = config.sample.num_batches_per_epoch
            training_batch_size = total_batch_size_local // num_batches
            samples_batched = []
            for k_batch in range(num_batches):
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                samples_batched.append({key: val[start:end] for key, val in shuffled.items()})

            info = defaultdict(list)
            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                cur_micro_bs = len(train_sample_batch["prompt_embeds"])

                # Prepare text embeddings (no CFG when guidance_scale <= 1.0)
                if config.sample.guidance_scale > 1.0:
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:cur_micro_bs], train_sample_batch["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:cur_micro_bs], train_sample_batch["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = train_sample_batch["prompt_embeds"]
                    pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

                for j_idx in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            x0 = train_sample_batch["latents_clean"]
                            t = train_sample_batch["timesteps"][:, j_idx] / 1000.0
                            t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))
                            noise = torch.randn_like(x0.float())
                            xt = (1 - t_expanded) * x0 + t_expanded * noise

                            # "old" adapter prediction (reference, no grad)
                            pipeline.transformer.set_adapter("old")
                            with torch.no_grad():
                                old_prediction = transformer(
                                    hidden_states=xt,
                                    timestep=train_sample_batch["timesteps"][:, j_idx],
                                    encoder_hidden_states=embeds,
                                    pooled_projections=pooled_embeds,
                                    return_dict=False,
                                )[0].detach()
                            pipeline.transformer.set_adapter("default")

                            # "default" adapter prediction (current policy, with grad)
                            forward_prediction = transformer(
                                hidden_states=xt,
                                timestep=train_sample_batch["timesteps"][:, j_idx],
                                encoder_hidden_states=embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0]

                            # Frozen base model prediction (ref, no grad)
                            with torch.no_grad():
                                with pipeline.transformer.disable_adapter():
                                    ref_forward_prediction = transformer(
                                        hidden_states=xt,
                                        timestep=train_sample_batch["timesteps"][:, j_idx],
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        return_dict=False,
                                    )[0]
                            # Restore to default after disable_adapter context
                            pipeline.transformer.set_adapter("default")

                            if config.train.ema_beta > 0:
                                pipeline.transformer.set_adapter("ema")
                                with torch.no_grad():
                                    ema_prediction = transformer(
                                        hidden_states=xt,
                                        timestep=train_sample_batch["timesteps"][:, j_idx],
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        return_dict=False,
                                    )[0]
                                pipeline.transformer.set_adapter("default")

                        # ---- NFT loss computation ----
                        advantages_clip = torch.clamp(
                            train_sample_batch["advantages"][:, j_idx],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )

                        # Apply advantage mode filtering
                        if hasattr(config.train, "adv_mode"):
                            if config.train.adv_mode == "positive_only":
                                advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
                            elif config.train.adv_mode == "negative_only":
                                advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
                            elif config.train.adv_mode == "one_only":
                                advantages_clip = torch.where(
                                    advantages_clip > 0,
                                    torch.ones_like(advantages_clip),
                                    torch.zeros_like(advantages_clip),
                                )
                            elif config.train.adv_mode == "binary":
                                advantages_clip = torch.sign(advantages_clip)

                        loss_terms = {}
                        loss_terms["x0_norm"] = torch.mean(x0 ** 2).detach()
                        loss_terms["x0_norm_max"] = torch.max(x0 ** 2).detach()
                        loss_terms["old_deviate"] = torch.mean(
                            (forward_prediction - old_prediction) ** 2
                        ).detach()
                        loss_terms["old_deviate_max"] = torch.max(
                            (forward_prediction - old_prediction) ** 2
                        ).detach()

                        if config.loss_type == "nft":
                            # Normalize advantage to [0, 1] for the NFT positive/negative blend.
                            normalized_adv = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5
                            r = torch.clamp(normalized_adv, 0, 1)

                            positive_prediction = (
                                config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
                            )
                            implicit_negative_prediction = (
                                (1.0 + config.beta) * old_prediction.detach() - config.beta * forward_prediction
                            )

                            x0_prediction = xt - t_expanded * positive_prediction
                            with torch.no_grad():
                                weight_factor = (
                                    torch.abs(x0_prediction.double() - x0.double())
                                    .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                                    .clip(min=0.00001)
                                )
                            positive_loss = (
                                (x0_prediction - x0) ** 2 / weight_factor
                            ).mean(dim=tuple(range(1, x0.ndim)))

                            negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
                            with torch.no_grad():
                                negative_weight_factor = (
                                    torch.abs(negative_x0_prediction.double() - x0.double())
                                    .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                                    .clip(min=0.00001)
                                )
                            negative_loss = (
                                (negative_x0_prediction - x0) ** 2 / negative_weight_factor
                            ).mean(dim=tuple(range(1, x0.ndim)))

                            ori_policy_loss = (
                                r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
                            )
                            policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()
                        else:
                            v_instant = noise - x0
                            advantage_weight = advantages_clip.view(-1, *([1] * (x0.ndim - 1)))
                            ori_policy_loss = (
                                advantage_weight * (forward_prediction - v_instant) ** 2
                            ).mean(dim=tuple(range(1, x0.ndim)))
                            policy_loss = ori_policy_loss.mean()
                        loss = policy_loss

                        loss_terms["policy_loss"] = policy_loss.detach()
                        loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()

                        # KL regularization against frozen base model
                        kl_div = (
                            (forward_prediction - ref_forward_prediction) ** 2
                        ).mean(dim=tuple(range(1, x0.ndim)))
                        loss = loss + config.train.beta * torch.mean(kl_div)
                        loss_terms["kl_div_loss"] = torch.mean(kl_div).detach()
                        loss_terms["kl_div"] = torch.mean(kl_div).detach()
                        loss_terms["old_kl_div"] = torch.mean(
                            ((old_prediction - ref_forward_prediction) ** 2).mean(
                                dim=tuple(range(1, x0.ndim))
                            )
                        ).detach()
                        if config.train.ema_beta > 0:
                            ema_kl_div = (
                                (forward_prediction - ema_prediction) ** 2
                            ).mean(dim=tuple(range(1, x0.ndim)))
                            loss = loss + config.train.ema_beta * torch.mean(ema_kl_div)
                            loss_terms["ema_kl_div_loss"] = torch.mean(ema_kl_div).detach()
                            loss_terms["ema_kl_div"] = torch.mean(ema_kl_div).detach()
                        loss_terms["total_loss"] = loss.detach()

                        for k_info, v_info in loss_terms.items():
                            info[k_info].append(v_info)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        info_reduced = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info_reduced = accelerator.reduce(info_reduced, reduction="mean")
                        info_reduced.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(
                                {
                                    **{
                                        f"train/{k_log}": (
                                            v_log.item()
                                            if isinstance(v_log, torch.Tensor)
                                            else v_log
                                        )
                                        for k_log, v_log in info_reduced.items()
                                    },
                                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                                },
                                step=global_step,
                                commit=True,
                            )

                        if config.train.ema and ema is not None:
                            ema.step(transformer_trainable_parameters, global_step)

                        global_step += 1
                        info = defaultdict(list)

        accelerator.wait_for_everyone()

        #################### DECAY UPDATE of "old" adapter ####################
        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(
                    tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay)
                )
            if config.train.ema_beta > 0:
                if config.train.kl_ema_decay_type == "constant":
                    ema_decay = config.train.kl_ema_decay
                elif config.train.kl_ema_decay_type == "linear":
                    ema_decay = min(config.train.kl_ema_decay, 0.001 * global_step)
                else:
                    raise ValueError(f"Unknown kl_ema_decay_type: {config.train.kl_ema_decay_type}")
                for src_param, tgt_param in zip(
                    transformer_trainable_parameters, ema_transformer_trainable_parameters, strict=True
                ):
                    tgt_param.data.copy_(
                        tgt_param.detach().data * ema_decay + src_param.detach().clone().data * (1.0 - ema_decay)
                    )

        if config.train.ema_beta > 0 and accelerator.is_main_process:
            wandb.log(
                {"train/ema_kl_decay": ema_decay},
                step=global_step,
                commit=False,
            )

        epoch += 1

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
