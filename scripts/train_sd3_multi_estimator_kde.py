from collections import defaultdict
import contextlib
import os
# Fix randomness for deterministic reproducibility
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import datetime
from concurrent import futures
import time
import json
import hashlib
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
# from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
# from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
# Fix randomness for deterministic reproducibility
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from flow_grpo.logging_utils import ExperimentLogger

wandb = ExperimentLogger()
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
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
import math
from typing import Optional, Union, Any, Dict, List
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

import sys
import shlex
import shutil

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: Optional[str] = 'sde',
    return_sqrt_dt: Optional[bool] = False,
    variance_noise: torch.FloatTensor = None,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma

    if sde_type == 'sde':
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level

        # our sde
        prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt

        if prev_sample is None:
            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

            sto_one_sample_estimator = torch.sqrt(-1*dt) * variance_noise / std_dev_t
        else:
            sto_one_sample_estimator = None

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
            # - torch.log(std_dev_t * torch.sqrt(-1*dt))
            # - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
    elif sde_type == 'dpm':
        sigma_max = 0.98 * torch.ones_like(sigma)
        sigma_VE = torch.where(sigma == 1, sigma_max, sigma) / (1 - torch.where(sigma == 1, sigma_max, sigma))
        sigma_VE_prev = sigma_prev / (1 - sigma_prev)
        eta = noise_level
        eps_pred = sample + (1 - sigma) * model_output
        # our sde
        sample_VE = sample * (1 + sigma_VE)
        prev_sample_mean_VE = sample_VE + (1 + eta) * eps_pred * (sigma_VE_prev - sigma_VE)
        prev_sample_mean = prev_sample_mean_VE / (1 + sigma_VE_prev)
        std_dev_t = torch.sqrt(eta * (sigma_VE**2 - sigma_VE_prev**2)) / (1 + sigma_VE_prev)
        if prev_sample is None:
            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
            prev_sample_VE = prev_sample_mean_VE + torch.sqrt(eta * (sigma_VE**2 - sigma_VE_prev**2)) * variance_noise
            prev_sample = prev_sample_VE / (1 + sigma_VE_prev)
        else:
            prev_sample_VE = prev_sample * (1 + sigma_VE_prev)
        log_prob = (
            -((prev_sample_VE.detach() - prev_sample_mean_VE) ** 2) / (2 * eta * (sigma_VE**2 - sigma_VE_prev**2))
            # - torch.log(std_dev_t * torch.sqrt(-1*dt))
            # - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

    elif sde_type == 'cps':
        std_dev_t = sigma_prev  * math.sin(noise_level * math.pi / 2) # sigma_t in paper
        pred_original_sample = sample - sigma * model_output # predicted x_0 in paper
        noise_estimate = sample + model_output * (1 - sigma) # predicted x_1 in paper
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)

        if prev_sample is None:
            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        # remove all constants
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    if return_sqrt_dt:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1*dt), sto_one_sample_estimator
    return prev_sample, log_prob, prev_sample_mean, std_dev_t

@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_layer_guidance_scale: float = 2.8,
    noise_level: float = 0.7,
    return_prev_sample_mean: bool = False,
    sde_type: Optional[str] = "sde",
    return_velocity: bool = False,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    ).float()

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_log_probs = []
    all_prev_latents_mean = []
    all_velocity = []
    all_noises = []
    all_sto_one_sample_estimator = []
    # 7. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            # noise_pred = noise_pred.to(prompt_embeds.dtype)
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            latents_dtype = latents.dtype

            variance_noise = randn_tensor(
                noise_pred.shape,
                device=noise_pred.device,
                dtype=(noise_pred.float()).dtype,
            )
            if return_velocity:
                latents, log_prob, prev_latents_mean, std_dev_t, sqrt_dt, sto_one_sample_estimator = sde_step_with_logprob(
                    self.scheduler, 
                    noise_pred.float(), 
                    t.unsqueeze(0), 
                    latents.float(),
                    noise_level=noise_level,
                    variance_noise=variance_noise,
                    sde_type=sde_type,
                    return_sqrt_dt=True,
                )
                all_velocity.append(noise_pred)
                all_noises.append(variance_noise)
                all_sto_one_sample_estimator.append(sto_one_sample_estimator)
            else:
                latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                    self.scheduler, 
                    noise_pred.float(), 
                    t.unsqueeze(0), 
                    latents.float(),
                    noise_level=noise_level,
                    variance_noise=variance_noise,
                )
            
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            all_prev_latents_mean.append(prev_latents_mean)
            # if latents.dtype != latents_dtype:
            #     latents = latents.to(latents_dtype)
            
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if return_prev_sample_mean:
        return image, all_latents, all_log_probs, all_velocity, all_noises, all_prev_latents_mean, all_sto_one_sample_estimator
    if return_velocity:
        return image, all_latents, all_log_probs, all_velocity, all_noises
    return image, all_latents, all_log_probs

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
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        
def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt, sto_one_sample_estimator = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
        sde_type=config.sde_type,
        return_sqrt_dt=True,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t, noise_pred, sqrt_dt, sto_one_sample_estimator

def diffusion_forward_with_cfg(transformer, pipeline, latent, timestep, embeds, pooled_embeds, config, mode="velocity"):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([latent] * 2),
            timestep=torch.cat([timestep] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=latent,
            timestep=timestep,
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    if mode=="velocity":
        return noise_pred
    elif mode=="epsilon":
        return latent + (1 - timestep) * noise_pred
    else:
        return 0

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
    elif weight_type == "nft":
        return -2.0 / config.nft_beta
    elif weight_type == "sto_uniform":
        return torch.sqrt(t / (1-torch.where(t == 1, t_j_max, t))) * config.sample.noise_level / sqrt_dt
    elif weight_type in ("det_t_exp_a", "t_exp_a"):
        return -2.0 * (t ** config.weight_exp)
    elif weight_type == "sto_t_exp_a":
        return torch.sqrt(t / (1-torch.where(t == 1, t_j_max, t))) * config.sample.noise_level / sqrt_dt * (t ** config.weight_exp)
    elif weight_type == "vanilla_var_reduction":
        w = -(1+config.sample.noise_level**2 / 2) * sqrt_dt**2 * (1-torch.where(t == 1, t_j_max, t)) / t
        return w
    elif weight_type == "grpo_guard":
        assert sqrt_dt > 0, "invalid sqrt_dt"
        w = (1+config.sample.noise_level**2 / 2) * config.sample.noise_level * torch.sqrt(t / (1-torch.where(t == 1, t_j_max, t))) / sqrt_dt
        return w
    
    else:
        raise NotImplementedError(f"Unknown weight type: {weight_type}")


def get_off_policy_weight(t, advantage, config, weight_type, t_j_max=1.0, sqrt_dt=1.0):
    if weight_type in ("sto_one_sample", "vanilla_var_reduction"):
        w = (1 + config.sample.noise_level**2 / 2)**2 / (2 * config.sample.noise_level**2)
        return advantage * w * sqrt_dt**2 * (1-torch.where(t == 1, t_j_max, t)) / t
    elif weight_type == "det_one_sample":
        return advantage
    elif weight_type == "nft":
        return 1.0
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

def get_kl_weight(t, config, weight_type):
    if weight_type == "1_t_exp_a":
        return t ** config.kl_weight_exp
    elif weight_type == "1_1t_exp_a":
        return (1.0 - t) ** config.kl_weight_exp
    else:
        raise NotImplementedError(f"Unknown weight type: {weight_type}")

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()
        offload_reward_fn = getattr(reward_fn, "offload", None)
        if offload_reward_fn is not None:
            offload_reward_fn()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
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
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
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

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


SD3_TRAINER_NAME = "train_sd3_multi_estimator_kde"
RESUME_MODEL_NAME = "resume_model.pt"
RESUME_OPTIMIZER_NAME = "resume_optimizer.pt"
RESUME_EMA_NAME = "resume_ema.pt"
RESUME_SCALER_NAME = "resume_scaler.pt"
RESUME_TRAINING_STATE_NAME = "resume_training_state.pt"


def resolve_resume_checkpoint(config, accelerator):
    expected = expected_metadata(
        config,
        trainer=SD3_TRAINER_NAME,
        world_size=accelerator.num_processes,
    )
    search_root = config.resume_from if config.resume_from else config.save_dir
    checkpoint_holder = [None]
    if accelerator.is_main_process:
        checkpoint_holder[0] = find_latest_checkpoint(
            search_root,
            expected,
            include_checkpoints_subdir=True,
        )
    if accelerator.num_processes > 1:
        torch.distributed.broadcast_object_list(checkpoint_holder, src=0)
    checkpoint = checkpoint_holder[0]
    if config.resume_from and checkpoint is None:
        raise ValueError(
            f"No compatible complete checkpoint found under {config.resume_from}"
        )
    return checkpoint


def load_resume_checkpoint(
    checkpoint_dir,
    transformer,
    optimizer,
    accelerator,
    ema,
    config,
    train_sampler,
):
    metadata = load_metadata(checkpoint_dir)
    expected = expected_metadata(
        config,
        trainer=SD3_TRAINER_NAME,
        world_size=accelerator.num_processes,
    )
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Checkpoint metadata mismatch for {key}: "
                f"checkpoint={metadata.get(key)!r}, current={value!r}"
            )

    model = unwrap_model(transformer, accelerator)
    model_state = torch_load(
        os.path.join(checkpoint_dir, RESUME_MODEL_NAME),
        map_location="cpu",
    )
    if config.use_lora:
        set_peft_model_state_dict(model, model_state, adapter_name="default")
    else:
        model.load_state_dict(model_state)

    optimizer.load_state_dict(
        torch_load(
            os.path.join(checkpoint_dir, RESUME_OPTIMIZER_NAME),
            map_location=accelerator.device,
        )
    )
    if config.train.ema:
        ema.load_state_dict(
            torch_load(
                os.path.join(checkpoint_dir, RESUME_EMA_NAME),
                map_location="cpu",
            )
        )
    scaler_path = os.path.join(checkpoint_dir, RESUME_SCALER_NAME)
    if accelerator.scaler is not None and os.path.isfile(scaler_path):
        accelerator.scaler.load_state_dict(
            torch_load(scaler_path, map_location="cpu")
        )

    rank_state_path = os.path.join(
        checkpoint_dir,
        f"{RESUME_TRAINING_STATE_NAME}.{accelerator.process_index}",
    )
    rank_state = torch_load(rank_state_path, map_location="cpu")
    if rank_state["process_index"] != accelerator.process_index:
        raise ValueError(f"Rank state mismatch in {rank_state_path}")
    train_sampler.set_epoch(rank_state["sampler_epoch"])
    accelerator.step = rank_state["accelerator_step"]
    load_rng_state_dict(rank_state["rng_state"])
    accelerator.wait_for_everyone()
    return rank_state["epoch"], rank_state["global_step"]


def save_ckpt(
    save_dir,
    transformer,
    global_step,
    accelerator,
    ema,
    transformer_trainable_parameters,
    config,
    optimizer,
    epoch,
    train_sampler,
):
    checkpoint_root = os.path.join(save_dir, "checkpoints")
    save_root = os.path.join(checkpoint_root, f"checkpoint-{global_step}")
    tmp_root = os.path.join(checkpoint_root, f".checkpoint-{global_step}.tmp")
    checkpoint_exists = [False]
    if accelerator.is_main_process:
        checkpoint_exists[0] = os.path.exists(save_root)
    if accelerator.num_processes > 1:
        torch.distributed.broadcast_object_list(checkpoint_exists, src=0)
    if checkpoint_exists[0]:
        raise ValueError(
            f"Checkpoint already exists and will not be overwritten: {save_root}"
        )

    if accelerator.is_main_process:
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root)
        os.makedirs(checkpoint_root, exist_ok=True)
    accelerator.wait_for_everyone()
    os.makedirs(tmp_root, exist_ok=True)

    save_root_lora = os.path.join(tmp_root, "lora")
    if accelerator.is_main_process:
        os.makedirs(save_root_lora, exist_ok=True)
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

        model = unwrap_model(transformer, accelerator)
        if config.use_lora:
            model_state = get_peft_model_state_dict(
                model,
                adapter_name="default",
            )
        else:
            model_state = model.state_dict()
        torch.save(model_state, os.path.join(tmp_root, RESUME_MODEL_NAME))
        torch.save(
            optimizer.state_dict(),
            os.path.join(tmp_root, RESUME_OPTIMIZER_NAME),
        )
        if config.train.ema:
            torch.save(ema.state_dict(), os.path.join(tmp_root, RESUME_EMA_NAME))
        if accelerator.scaler is not None:
            torch.save(
                accelerator.scaler.state_dict(),
                os.path.join(tmp_root, RESUME_SCALER_NAME),
            )
        metadata = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "sampler_epoch": int(train_sampler.epoch),
            **expected_metadata(
                config,
                trainer=SD3_TRAINER_NAME,
                world_size=accelerator.num_processes,
            ),
        }
        with open(
            os.path.join(tmp_root, META_NAME),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2, sort_keys=True)

    accelerator.wait_for_everyone()
    rank_state = {
        "accelerator_step": int(accelerator.step),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "process_index": int(accelerator.process_index),
        "rng_state": rng_state_dict(),
        "sampler_epoch": int(train_sampler.epoch),
    }
    torch.save(
        rank_state,
        os.path.join(
            tmp_root,
            f"{RESUME_TRAINING_STATE_NAME}.{accelerator.process_index}",
        ),
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
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
    accelerator.wait_for_everyone()
    return save_root

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
        param_group['lr'] = lr
    return lr

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        if not config.train.cfg:
            config.run_name += "_wotcfg"
        if config.sample.guidance_scale == 1.0:
            config.run_name += '_woscfg'
        

        config.run_name += f"_noise{config.sample.noise_level:.2f}"

        if config.sample.num_steps > 10:
            config.run_name += f"_sample{config.sample.num_steps}"
        
        if config.sample.same_latent:
            config.run_name += f"_samelatents"
        
        if config.policy_gradient_weight in ("det_t_exp_a", "t_exp_a"):
            config.run_name += f"_w2_exp_{config.weight_exp}"
        
        

        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * (config.end_timesteps - config.start_timesteps) #num_train_timesteps,
    )
    if accelerator.is_main_process:
        wandb.init(
            project="diffusion-rl",
            name=config.run_name,
            config=config.to_dict(),
        )
        logger.info(f"\n{config}")

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

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        # Set correct lora layers
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
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

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

    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, "multi_score")(
        accelerator.device,
        config.reward_fn,
        config.reward_backends,
        hpsv3_offload=config.hpsv3_offload,
    )
    if "hpsv3" in config.reward_fn and not config.hpsv3_offload:
        eval_reward_fn = reward_fn
    else:
        eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(
            accelerator.device,
            config.reward_fn,
            config.reward_backends,
            hpsv3_offload=config.hpsv3_offload,
        )
    train_dataloader_generator = torch.Generator().manual_seed(
        config.seed + accelerator.process_index
    )

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
            collate_fn=TextPromptDataset.collate_fn,
            generator=train_dataloader_generator,
            # persistent_workers=True
        )

        # Create a regular DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
            collate_fn=GenevalPromptDataset.collate_fn,
            generator=train_dataloader_generator,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")


    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)

    resume_checkpoint = resolve_resume_checkpoint(config, accelerator)
    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)
    if resume_checkpoint is not None:
        logger.info(f"Resuming training from {resume_checkpoint}")
        epoch, global_step = load_resume_checkpoint(
            resume_checkpoint,
            transformer,
            optimizer,
            accelerator,
            ema,
            config,
            train_sampler,
        )
    set_lr(optimizer, config, global_step)
    last_checkpoint_step = global_step if resume_checkpoint is not None else -1

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
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

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    while epoch < config.num_epochs:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if config.eval and epoch % config.eval_freq == 0:
            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    images, latents, log_probs, velocities, noises, prev_sample_mean, sto_one_sample_estimators = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        generator=generator,
                        sde_type=config.sde_type,
                        return_velocity=True,
                        return_prev_sample_mean=True,
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            prev_sample_mean = torch.stack(
                prev_sample_mean, dim=1
            )
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
            velocities = torch.stack(velocities, dim=1)
            sto_one_sample_estimators = torch.stack(sto_one_sample_estimators, dim=1)
            noises = torch.stack(noises, dim=1)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)
            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "clean_latents": latents[:, -1],
                    "det_one_sample_estimator": (latents[:,:-1] - latents[:, -1].unsqueeze(1)) / (timesteps.view(timesteps.shape[0],timesteps.shape[1],1,1,1) / 1000.0),
                    "sto_one_sample_estimator": sto_one_sample_estimators,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                    "velocities": velocities,
                    "noises": noises,
                    "prev_sample_mean": prev_sample_mean,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
        offload_reward_fn = getattr(reward_fn, "offload", None)
        if offload_reward_fn is not None:
            offload_reward_fn()

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # Use the new index

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

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
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
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

        gathered_prompt_ids = accelerator.gather(samples["prompt_ids"])
        prompts = pipeline.tokenizer.batch_decode(
            gathered_prompt_ids.cpu().numpy(), skip_special_tokens=True
        )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
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
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )

        kde_timesteps = [
            step_index
            for step_index in range(config.start_timesteps, config.end_timesteps)
            if get_timestep_config_value(config.value_gradient_estimator, step_index) == "kde"
        ]
        use_kde = len(kde_timesteps) > 0
        if use_kde:
            with torch.no_grad():
                prompt_to_group = {prompt: group_id for group_id, prompt in enumerate(dict.fromkeys(prompts))}
                prompt_group_ids = torch.as_tensor(
                    [prompt_to_group[prompt] for prompt in prompts],
                    device=accelerator.device,
                )
                local_prompt_group_ids = prompt_group_ids.reshape(accelerator.num_processes, -1)[accelerator.process_index]
                kde_estimator, kde_stats = compute_kde_estimator(
                    local_prompt_group_ids=local_prompt_group_ids,
                    local_latents=samples["latents"],
                    local_velocities=samples["velocities"],
                    local_timesteps=samples["timesteps"],
                    global_prompt_group_ids=prompt_group_ids,
                    global_clean_latents=accelerator.gather(samples["clean_latents"]),
                    global_advantages=accelerator.gather(samples["advantages"]),
                    adv_clip_max=config.train.adv_clip_max,
                    step_indices=kde_timesteps,
                    kde_bandwidth=config.kde_bandwidth,
                )
                samples["kde_estimator"] = kde_estimator
                gathered_kde_stats = {
                    stat_name: accelerator.gather(stat_values)
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
                    if accelerator.is_main_process:
                        kde_log[base_tag] = overall_mean.item()

                    for stats_index, step_index in enumerate(kde_timesteps):
                        timestep_tag = f"{base_tag}/timestep_{step_index}"
                        timestep_mean = stat_values[:, stats_index].mean()
                        if accelerator.is_main_process:
                            kde_log[timestep_tag] = timestep_mean.item()
                if accelerator.is_main_process:
                    wandb.log(kde_log, step=global_step, commit=False)

        del samples["rewards"]
        del samples["prompt_ids"]

        num_batches = config.sample.num_batches_per_epoch
        if use_kde:
            if accelerator.is_main_process:
                wandb.log(
                    {
                        "train/actual_batch_size": samples["advantages"].shape[0]
                        // num_batches
                    },
                    step=global_step,
                    commit=False,
                )
        else:
            # Get the mask for samples where all advantages are zero across the time dimension
            mask = (samples["advantages"].abs().sum(dim=1) != 0)

            # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
            # randomly change some False values to True to make it divisible
            true_count = mask.sum()
            if true_count % num_batches != 0:
                false_indices = torch.where(~mask)[0]
                num_to_change = num_batches - (true_count % num_batches)
                if len(false_indices) >= num_to_change:
                    random_indices = torch.randperm(len(false_indices))[:num_to_change]
                    mask[false_indices[random_indices]] = True
            if accelerator.is_main_process:
                wandb.log(
                    {"train/actual_batch_size": mask.sum().item() // num_batches},
                    step=global_step,
                    commit=False,
                )
            # Filter out samples where the entire time dimension of advantages is zero
            samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        # assert (
        #     total_batch_size
        #     == config.sample.train_batch_size * config.sample.num_batches_per_epoch
        # )
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                gradient_train_timesteps = list(range(config.start_timesteps, config.end_timesteps))
                for j in tqdm(
                    gradient_train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t, velocity, sqrt_dt, sto_one_sample_estimator = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _, velocity_ref, sqrt_dt, sto_one_sample_estimator = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        t_j = sample["timesteps"][:,j].mean() / 1000.0
                        t_j_max = sample["timesteps"][:,1].mean() / 1000.0

                        value_grad_estimator_type = get_timestep_config_value(config.value_gradient_estimator, j)
                        if value_grad_estimator_type == "sto_one_sample":
                            value_grad_estimator = advantages.view([std_dev_t.shape[0],1,1,1]) * sample["sto_one_sample_estimator"][:, j]
                        elif value_grad_estimator_type == "det_one_sample":
                            value_grad_estimator = advantages.view([std_dev_t.shape[0],1,1,1]) * (sample["det_one_sample_estimator"][:, j] - sample["velocities"][:, j])
                        elif value_grad_estimator_type == "kde":
                            value_grad_estimator = sample["kde_estimator"][:, j]
                        else:
                            raise NotImplementedError(f"Unknown value-gradient estimator: {value_grad_estimator_type}")

                        off_policy_weight_type = get_timestep_config_value(config.off_policy_weight, j)
                        off_policy_weight = get_off_policy_weight(t_j, advantages, config, off_policy_weight_type, t_j_max=t_j_max, sqrt_dt=sqrt_dt.mean())
                        
                        policy_gradient_weight_type = get_timestep_config_value(config.policy_gradient_weight, j)
                        policy_gradient_weight = get_policy_gradient_weight(t_j, config, policy_gradient_weight_type, t_j_max=t_j_max, sqrt_dt=sqrt_dt.mean())
                        off_policy_loss = off_policy_weight * ((velocity - sample["velocities"][:, j])**2).mean(dim=(1,2,3)) 
                        policy_gradient_loss = policy_gradient_weight * ((velocity - sample["velocities"][:, j]) * value_grad_estimator).mean(dim=(1,2,3))

                        clipped_loss = (policy_gradient_loss + off_policy_loss)
                        policy_loss = torch.mean(clipped_loss)

                        ######### old implementation #########
                        # # check grpo implementation
                        # sigma_t = std_dev_t.mean()
                        # ratio_mean_bias = (prev_sample_mean.detach() - sample["prev_sample_mean"][:, j]).pow(2).mean(dim=(1,2,3))
                        # ratio_mean_bias = ratio_mean_bias / (2 * (sqrt_dt.mean() * sigma_t) ** 2)
                        # log_ratio = (log_prob - sample["log_probs"][:, j] + ratio_mean_bias) #* (sqrt_dt.mean() * sigma_t)
                        # # check awm
                        # v_instant = (sample["latents"][:, j] - sample["next_latents"][:, -1]) / t_j
                        # loss_v = (advantages.view([std_dev_t.shape[0],1,1,1]) * ((velocity - v_instant) ** 2)).mean(dim=(1,2,3))
                        
                        if config.train.beta > 0:
                            #kl_loss_first_term = total_ratio.view([std_dev_t.shape[0],1,1,1]) * ((velocity - velocity_ref) ** 2).mean(dim=(1,2,3), keepdim=True) * delta_lambda_j.view([std_dev_t.shape[0],1,1,1])
                            if config.reweight_kl:
                                kl_loss = ((velocity - velocity_ref) ** 2).mean(dim=(1,2,3), keepdim=True)
                                if config.kl_weight_type:
                                    kl_loss_weight = get_kl_weight(t_j, config, config.kl_weight_type)
                                else:
                                    kl_loss_weight = 1.0
                                kl_loss = kl_loss * kl_loss_weight
                                kl_loss_first_term = kl_loss
                                kl_loss_second_term = kl_loss
                            else:
                                kl_loss_first_term = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                                kl_loss_second_term =  total_kl_loss * log_ratio.view([std_dev_t.shape[0],1,1,1])
                                # kl_loss_second_term = 0
                                kl_loss = (kl_loss_first_term + kl_loss_second_term) # * mask.view([std_dev_t.shape[0],1,1,1])
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        # info["total_ratio"].append(
                        #     torch.mean(total_ratio)
                        # )
                        # info["clipfrac"].append(
                        #     torch.mean(
                        #         (
                        #             torch.abs(total_ratio - 1.0) > config.train.clip_range
                        #         ).float()
                        #     )
                        # )
                        # info["clipfrac_gt_one"].append(
                        #     torch.mean(
                        #         (
                        #             total_ratio - 1.0 > config.train.clip_range
                        #         ).float()
                        #     )
                        # )
                        # info["clipfrac_lt_one"].append(
                        #     torch.mean(
                        #         (
                        #             1.0 - total_ratio > config.train.clip_range
                        #         ).float()
                        #     )
                        # )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)
                            # info["kl_loss_first_term"].append(torch.mean(kl_loss_first_term * mask.view([std_dev_t.shape[0],1,1,1])))
                            # info["kl_loss_second_term"].append(torch.mean(kl_loss_second_term * mask.view([std_dev_t.shape[0],1,1,1])))

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
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
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
        
        epoch += 1
        if (
            config.save
            and (epoch % config.save_freq == 0 or epoch == config.num_epochs)
            and global_step > last_checkpoint_step
        ):
            if not accelerator.sync_gradients:
                raise RuntimeError(
                    "Cannot save a resumable checkpoint with pending accumulated gradients"
                )
            save_ckpt(
                config.save_dir,
                transformer,
                global_step,
                accelerator,
                ema,
                transformer_trainable_parameters,
                config,
                optimizer,
                epoch,
                train_sampler,
            )
            last_checkpoint_step = global_step

    if accelerator.is_main_process:
        wandb.finish()
        
if __name__ == "__main__":
    app.run(main)

