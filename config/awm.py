import importlib.util
import os
import datetime


def _load_base():
    spec = importlib.util.spec_from_file_location("base", os.path.join(os.path.dirname(__file__), "base.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()


def get_config(name):
    return globals()[name]()


def _get_base():
    config = base.get_config()
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.resolution = 512
    config.use_lora = True
    config.num_epochs = 700
    config.sample.num_steps = 14
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0
    config.sample.num_image_per_prompt = 24
    config.per_prompt_stat_tracking = True
    config.sample.global_std = True
    config.train.cfg = False
    return config


def sd3_geneval_1node():
    config = _get_base()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    gpu_numbers = 8
    config.sample.train_batch_size = 6
    config.sample.num_batches_per_epoch = 24 * 24 // gpu_numbers // 2
    config.sample.test_batch_size = 6
    config.sample.noise_level = 0.4
    config.sample.sde_frac = 0.0

    config.train.batch_size = 3
    config.train.gradient_accumulation_steps = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_batches_per_epoch = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_inner_epochs = 1
    config.train.beta = 0.003
    config.train.train_timesteps = 6
    config.train.kl_weight = 'Uniform'
    config.train.advantage_max = 1
    config.train.ema = True
    config.train.timestep_fraction = 0.9
    config.train.loss_type = 'exp_first'
    config.train.learning_rate = 3e-4
    config.train.weighting = 'ghuber'
    config.train.ghuber_power = 0.25
    config.train.kl_ema_weight = 'Uniform'
    config.train.kl_ema_decay = 0.3
    config.train.ema_beta = 1
    config.train.kl_ema_decay_type = 'linear'
    config.train.off_policy = False
    config.train.substitute_base_with_ema_long = False
    config.train.use_sa_solver = True
    config.train.cfg = False
    config.train.clip_range = 1
    config.train.decay_steps = 10000000000
    config.train.ema_decay = 0.99
    config.train.ema_update_step_interval = 1

    config.save_freq = 15
    config.eval_freq = 15
    config.reward_fn = {"geneval": 1.0}
    config.prompt_fn = "geneval"
    config.lora_type = "attn"
    config.time_type = "discrete_wo_init"
    config.time_shift = 3.0

    config.run_name = "awm_sd3_geneval_1node"
    config.save_dir = f"logs/awm/sd3/geneval"
    return config

def sd3_ocr_1node():
    config = _get_base()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    gpu_numbers = 8
    config.sample.train_batch_size = 6
    config.sample.num_batches_per_epoch = 24 * 24 // gpu_numbers // 2
    config.sample.test_batch_size = 6
    config.sample.noise_level = 0.4
    config.sample.sde_frac = 0.0

    config.train.batch_size = 3
    config.train.gradient_accumulation_steps = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_batches_per_epoch = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_inner_epochs = 1
    config.train.beta = 0.003
    config.train.train_timesteps = 6
    config.train.kl_weight = 'Uniform'
    config.train.advantage_max = 1
    config.train.ema = True
    config.train.timestep_fraction = 0.9
    config.train.loss_type = 'exp_first'
    config.train.learning_rate = 3e-4
    config.train.weighting = 'ghuber'
    config.train.ghuber_power = 0.25
    config.train.kl_ema_weight = 'Uniform'
    config.train.kl_ema_decay = 0.3
    config.train.ema_beta = 1
    config.train.kl_ema_decay_type = 'linear'
    config.train.off_policy = False
    config.train.substitute_base_with_ema_long = False
    config.train.use_sa_solver = True
    config.train.cfg = False
    config.train.clip_range = 1
    config.train.decay_steps = 10000000000
    config.train.ema_decay = 0.99
    config.train.ema_update_step_interval = 1

    config.save_freq = 15
    config.eval_freq = 15
    config.reward_fn = {"ocr": 1.0}
    config.prompt_fn = "general_ocr"
    config.lora_type = "attn"
    config.time_type = "discrete_wo_init"
    config.time_shift = 3.0

    config.run_name = "awm_sd3_ocr_1node"
    config.save_dir = f"logs/awm/sd3/ocr"
    return config

def sd3_pickscore_1node():
    config = _get_base()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    gpu_numbers = 8
    config.sample.train_batch_size = 6
    config.sample.num_batches_per_epoch = 8 * 24 // gpu_numbers // 2
    config.sample.test_batch_size = 6
    config.sample.noise_level = 0.0
    config.sample.sde_frac = 0.0

    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_batches_per_epoch = config.sample.train_batch_size // config.train.batch_size * config.sample.num_batches_per_epoch
    config.train.num_inner_epochs = 1
    config.train.beta = 0.0001
    config.train.train_timesteps = 6
    config.train.kl_weight = 'Uniform'
    config.train.advantage_max = 1
    config.train.ema = True
    config.train.timestep_fraction = 0.9
    config.train.loss_type = 'exp_first'
    config.train.learning_rate = 3e-4
    config.train.weighting = 'ghuber'
    config.train.ghuber_power = 0.25
    config.train.kl_ema_weight = 'Uniform'
    config.train.kl_ema_decay = 0.3
    config.train.ema_beta = 1
    config.train.kl_ema_decay_type = 'linear'
    config.train.off_policy = False
    config.train.substitute_base_with_ema_long = False
    config.train.use_sa_solver = True
    config.train.cfg = False
    config.train.clip_range = 1
    config.train.decay_steps = 10000000000
    config.train.ema_decay = 0.99
    config.train.ema_update_step_interval = 1

    config.save_freq = 15
    config.eval_freq = 15
    config.reward_fn = {"pickscore": 1.0}
    config.prompt_fn = "general_ocr"
    config.lora_type = "attn"
    config.time_type = "discrete"
    config.time_shift = 3.0

    config.run_name = "awm_sd3_pickscore_1node"
    config.save_dir = f"logs/awm/sd3/pickscore"
    return config