import importlib.util
import os
import ml_collections


def _load_base():
    spec = importlib.util.spec_from_file_location("base", os.path.join(os.path.dirname(__file__), "base.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()


def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True

    config.loss_type = "value_gradient"
    config.kl_weight_type = None

    return config


# ==============================================================================
# 1. Base / Template Configs for SD3.5-M PickScore (Sec. 4.2)
# ==============================================================================

# Paper reference: Sec. 4.2, Fig. 2(a, d) — Base 8-GPU template for the stochastic 1-sample estimator on SD3.5-M PickScore
def sd3_pickscore_sto_base(gradient_step_per_epoch=1):
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.logdir = "logs/"
    config.run_name = "sd3_pickscore_sto"

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % gradient_step_per_epoch == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16  # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // gradient_step_per_epoch
    config.num_epochs = 700 // gradient_step_per_epoch
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 0.05
    # kl loss
    config.train.beta = 0.0
    config.train.learning_rate = 3e-4
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.sample.noise_level = 0.7
    config.sde_type = "sde"
    config.sto_term = False
    config.det_term = False
    config.reweight_kl = True
    config.train.ema = True
    config.save_freq = 15  # epoch
    config.eval_freq = 15
    config.save_dir = "logs/pickscore/sd3.5-M"
    config.reward_fn = {
        "pickscore": 1.0,
    }
    config.prompt_fn = "general_ocr"
    config.start_timesteps = 0
    config.end_timesteps = 4
    config.per_prompt_stat_tracking = True
    return config


# Paper reference: Sec. 4.2, Fig. 2(b, e) — Base 8-GPU template for the deterministic 1-sample estimator on SD3.5-M PickScore
def sd3_pickscore_det_base(gradient_step_per_epoch=1):
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.logdir = "logs/"
    config.run_name = "sd3_pickscore_det"

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % gradient_step_per_epoch == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16  # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // gradient_step_per_epoch
    config.num_epochs = 700 // gradient_step_per_epoch
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 0.2
    # kl loss
    config.train.beta = 0.0
    config.train.learning_rate = 3e-4
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.sample.noise_level = 0.1
    config.sde_type = "sde"
    config.sto_term = False
    config.det_term = False
    config.reweight_kl = True
    config.train.ema = True
    config.save_freq = 15  # epoch
    config.eval_freq = 15
    config.save_dir = "logs/pickscore/sd3.5-M"
    config.reward_fn = {
        "pickscore": 1.0,
    }
    config.prompt_fn = "general_ocr"
    config.start_timesteps = 0
    config.end_timesteps = 4
    config.per_prompt_stat_tracking = True
    return config


# ==============================================================================
# 2. Deterministic & Stochastic 1-Sample Estimator Ablations (Sec. 4.2, Fig. 2 & Appx. G.1, Fig. 5)
# ==============================================================================

# Paper reference: Sec. 4.2, Fig. 2(b, e) & Appx. G.1, Fig. 5(b) — Deterministic 1-sample estimator with w2(t)=t^7, w1(t)=(1-t)^3, noise_level=0.1
def sd3_pickscore_det_w7_offw3():
    config = sd3_pickscore_det_base(gradient_step_per_epoch=2)
    config.run_name += "_w7_offw3_kl0001"
    config.sample.guidance_scale = 1.0
    config.train.cfg = False
    config.end_timesteps = 9
    config.value_gradient_estimator = "det_one_sample"  # ["det_one_sample", "sto_one_sample", "kde"]
    config.off_policy_weight = "1_1t_exp_a"
    config.policy_gradient_weight = "det_t_exp_a"
    config.weight_exp = 7.0
    config.off_policy_weight_exp = 3.0
    config.train.beta = 0.0001
    return config


# Paper reference: Sec. 4.2, Fig. 2(b, e) & Appx. G.1, Fig. 5(b) — Deterministic 1-sample estimator with smaller sampler noise (noise_level=0.05)
def sd3_pickscore_det_w7_offw3_noise005():
    config = sd3_pickscore_det_base(gradient_step_per_epoch=2)
    config.run_name += "_w7_offw3_noise005_kl0001"
    config.sample.guidance_scale = 1.0
    config.train.cfg = False
    config.end_timesteps = 9
    config.value_gradient_estimator = "det_one_sample"
    config.off_policy_weight = "1_1t_exp_a"
    config.policy_gradient_weight = "det_t_exp_a"
    config.weight_exp = 7.0
    config.off_policy_weight_exp = 3.0
    config.train.beta = 0.0001
    config.sample.noise_level = 0.05
    return config


# Paper reference: Sec. 4.2, Fig. 2(b, e) & Appx. G.1, Fig. 5(b) — Deterministic 1-sample estimator with larger sampler noise (noise_level=0.2)
def sd3_pickscore_det_w7_offw3_noise02():
    config = sd3_pickscore_det_base(gradient_step_per_epoch=2)
    config.run_name += "_w7_offw3_noise02_kl0001"
    config.sample.guidance_scale = 1.0
    config.train.cfg = False
    config.end_timesteps = 9
    config.value_gradient_estimator = "det_one_sample"
    config.off_policy_weight = "1_1t_exp_a"
    config.policy_gradient_weight = "det_t_exp_a"
    config.weight_exp = 7.0
    config.off_policy_weight_exp = 3.0
    config.train.beta = 0.0001
    config.sample.noise_level = 0.2
    return config


# Paper reference: Sec. 4.2, Fig. 2(b, e) — Deterministic 1-sample estimator with step-halving learning rate decay at steps (300, 600)
def sd3_pickscore_det_w7_offw3_lrdecay():
    config = sd3_pickscore_det_base(gradient_step_per_epoch=2)
    config.run_name += "_w7_offw3_kl0001_half300"
    config.sample.guidance_scale = 1.0
    config.train.cfg = False
    config.end_timesteps = 9
    config.value_gradient_estimator = "det_one_sample"
    config.off_policy_weight = "1_1t_exp_a"
    config.policy_gradient_weight = "det_t_exp_a"
    config.weight_exp = 7.0
    config.off_policy_weight_exp = 3.0
    config.train.beta = 0.0001
    config.train.lr_scheduler = "step_halving"
    config.train.lr_halving_steps = (300, 600)
    return config


# Paper reference: Sec. 4.2, Fig. 2(a, d) — Stochastic 1-sample estimator with optimal weight w2(t)=t^1, w1(t)=0, and KL beta=1e-4
def sd3_pickscore_sto_w1():
    config = sd3_pickscore_sto_base(gradient_step_per_epoch=2)
    config.run_name += "_w1_offw0_kl0001"
    config.sample.guidance_scale = 1.0
    config.train.cfg = False
    config.end_timesteps = 9
    config.value_gradient_estimator = "sto_one_sample"
    config.off_policy_weight = "None"
    config.policy_gradient_weight = "sto_t_exp_a"
    config.weight_exp = 1.0
    config.off_policy_weight_exp = 1.0
    config.train.beta = 0.0001
    return config


# Paper reference: Sec. 4.2, Fig. 2(a, d) — Stochastic 1-sample estimator with w2(t)=t^3, w1(t)=0, and shared initial latent noise per prompt
def sd3_pickscore_sto_w3_samenoise():
    config = sd3_pickscore_sto_base(gradient_step_per_epoch=2)
    config.run_name += "_w3_offw0_samenoise"
    config.sample.guidance_scale = 1.0
    config.train.cfg = False
    config.sample.same_latent = True
    config.end_timesteps = 9
    config.value_gradient_estimator = "sto_one_sample"
    config.off_policy_weight = "None"
    config.policy_gradient_weight = "sto_t_exp_a"
    config.weight_exp = 3.0
    config.off_policy_weight_exp = 1.0
    return config


# Paper reference: Sec. 4.2, Fig. 2(a, d) — Stochastic 1-sample estimator with w2(t)=t^3, w1(t)=0, without KL regularization
def sd3_pickscore_sto_w3_nokl():
    config = sd3_pickscore_sto_base(gradient_step_per_epoch=2)
    config.run_name += "_w3_offw0_nokl"
    config.sample.guidance_scale = 1.0
    config.train.cfg = False
    config.end_timesteps = 9
    config.value_gradient_estimator = "sto_one_sample"
    config.off_policy_weight = "None"
    config.policy_gradient_weight = "sto_t_exp_a"
    config.weight_exp = 3.0
    config.off_policy_weight_exp = 1.0
    return config


# Paper reference: Sec. 4.2, Fig. 2(a, d) — Stochastic 1-sample estimator with w2(t)=t^3, w1(t)=0, and KL beta=1e-4
def sd3_pickscore_sto_w3():
    config = sd3_pickscore_sto_base(gradient_step_per_epoch=2)
    config.run_name += "_w3_offw0_kl0001"
    config.sample.guidance_scale = 1.0
    config.train.cfg = False
    config.end_timesteps = 9
    config.value_gradient_estimator = "sto_one_sample"
    config.off_policy_weight = "None"
    config.policy_gradient_weight = "sto_t_exp_a"
    config.weight_exp = 3.0
    config.off_policy_weight_exp = 1.0
    config.train.beta = 0.0001
    return config


# ==============================================================================
# 3. SD3.5-M + PickScore with Multi-Sample KDE Estimator (Sec. 4.2, Fig. 2; Sec. 4.3, Fig. 3(b) & Fig. 4)
# ==============================================================================

# Paper reference: Sec. 4.2 & Sec. 4.3 — Base 8-GPU template for the multi-sample KDE estimator on SD3.5-M PickScore
def sd3_pickscore_kde_base(gradient_step_per_epoch=2):
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.logdir = "logs/"

    if gradient_step_per_epoch == 1:
        config.run_name = "sd3_pickscore_kde_onpolicy"
    else:
        config.run_name = "sd3_pickscore_kde_offpolicy"

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % gradient_step_per_epoch == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16  # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // gradient_step_per_epoch
    config.num_epochs = 700 // gradient_step_per_epoch
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 0.05
    # kl loss
    config.train.beta = 0
    config.train.learning_rate = 3e-4
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.sample.noise_level = 0.1
    config.sde_type = "sde"
    config.sto_term = False
    config.det_term = False
    config.reweight_kl = True
    config.kl_weight_type = None
    config.train.ema = True
    config.save_freq = 15  # epoch
    config.eval_freq = 15
    config.save_dir = "logs/pickscore/sd3.5-M/"
    config.reward_fn = {
        "pickscore": 1.0,
    }

    config.prompt_fn = "general_ocr"
    config.start_timesteps = 0
    config.end_timesteps = 9
    config.per_prompt_stat_tracking = True
    return config


# Paper reference: Sec. 4.2, Fig. 2(c) & Sec. 4.3, Fig. 4 — Multi-sample KDE estimator (on-policy, w2(t)=t^3, "+ our on-policy weight" ablation)
def sd3_pickscore_kde_onpolicy():
    config = sd3_pickscore_kde_base(gradient_step_per_epoch=1)
    config.train.cfg = False
    config.run_name += "_w3_m1000_kl0001_half235"
    config.save_dir += config.run_name
    config.sample.guidance_scale = 1.0
    config.sample.noise_level = 0.1
    config.kde_bandwidth = 16 * 64 * 64 / 1000.0

    config.value_gradient_estimator = "kde"
    config.off_policy_weight = "1_1t_exp_a"
    config.policy_gradient_weight = "det_t_exp_a"
    config.weight_exp = 3.0
    config.off_policy_weight_exp = 5.0
    config.train.beta = 0.0001
    config.train.lr_scheduler = "step_halving"
    config.train.lr_halving_steps = (200, 300, 500)
    return config


# Paper reference: Sec. 4.2, Fig. 2(f); Sec. 4.3, Fig. 3(b) & Fig. 4; Appx. G.3, Table 3 — Final KDE recipe on SD3.5-M PickScore (off-policy, w2(t)=t^3, w1(t)=(1-t)^5)
def sd3_pickscore_kde():
    config = sd3_pickscore_kde_base(gradient_step_per_epoch=2)
    config.train.cfg = False
    config.run_name += "_w3_offw5_m1000_kl0001_half235"
    config.save_dir += config.run_name
    config.sample.guidance_scale = 1.0
    config.sample.noise_level = 0.1
    config.kde_bandwidth = 16 * 64 * 64 / 1000.0

    config.value_gradient_estimator = "kde"
    config.off_policy_weight = "1_1t_exp_a"
    config.policy_gradient_weight = "det_t_exp_a"
    config.weight_exp = 3.0
    config.off_policy_weight_exp = 5.0
    config.train.beta = 0.0001
    config.train.lr_scheduler = "step_halving"
    config.train.lr_halving_steps = (200, 300, 500)
    return config


# ==============================================================================
# 4. SD3.5-M + OCR with Multi-Sample KDE Estimator (Sec. 4.3, Fig. 3(a))
# ==============================================================================

# Paper reference: Sec. 4.3, Fig. 3(a) — Base 8-GPU template for the multi-sample KDE estimator on SD3.5-M OCR
def sd3_ocr_kde_base(gradient_step_per_epoch=2):
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.logdir = "logs/"

    if gradient_step_per_epoch == 1:
        config.run_name = "sd3_ocr_kde_onpolicy"
    else:
        config.run_name = "sd3_ocr_kde_offpolicy"

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % gradient_step_per_epoch == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16  # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // gradient_step_per_epoch
    config.num_epochs = 100 // gradient_step_per_epoch
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 0.05
    # kl loss
    config.train.beta = 0
    config.train.learning_rate = 3e-4
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.sample.noise_level = 0.1
    config.sde_type = "sde"
    config.sto_term = False
    config.det_term = False
    config.reweight_kl = True
    config.train.ema = True
    config.save_freq = 15  # epoch
    config.eval_freq = 15
    config.save_dir = "logs/ocr-kde/sd3.5-M/"
    config.reward_fn = {
        "ocr": 1.0,
    }

    config.prompt_fn = "general_ocr"
    config.start_timesteps = 0
    config.end_timesteps = 9
    config.per_prompt_stat_tracking = True
    return config


# Paper reference: Sec. 4.3, Fig. 3(a) — Final multi-sample KDE recipe on SD3.5-M OCR (off-policy, w2(t)=t^7, w1(t)=(1-t)^3)
def sd3_ocr_kde():
    config = sd3_ocr_kde_base(gradient_step_per_epoch=2)
    config.train.cfg = False
    config.run_name += "_w7_offw3_m1000_kl0001"
    config.save_dir += config.run_name
    config.sample.guidance_scale = 1.0
    config.sample.noise_level = 0.1
    config.kde_bandwidth = 16 * 64 * 64 / 1000.0

    config.value_gradient_estimator = "kde"
    config.off_policy_weight = "1_1t_exp_a"
    config.policy_gradient_weight = "det_t_exp_a"
    config.weight_exp = 7.0
    config.off_policy_weight_exp = 3.0
    config.train.beta = 0.0001
    return config


# ==============================================================================
# 5. SD3.5-M + GenEval with Multi-Sample KDE Estimator (Sec. 4.3, Fig. 3(c))
# ==============================================================================

# Paper reference: Sec. 4.3, Fig. 3(c) — Final multi-sample KDE recipe on SD3.5-M GenEval (off-policy, w2(t)=t^1, w1(t)=0.05(1-t)^2)
def sd3_geneval_kde():
    config = sd3_pickscore_kde_base(gradient_step_per_epoch=2)
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")
    config.run_name = "sd3_geneval_kde"
    config.logdir = "logs/geneval"
    config.save_dir = os.path.join(config.logdir, "ckpt", config.run_name)
    config.num_epochs = 350
    config.resume_from = None
    config.eval_freq = 15
    config.save_freq = 15
    config.eval = True
    config.save = True
    config.prompt_fn = "geneval"
    config.reward_fn = {"geneval": 1.0}

    config.sample.test_batch_size = 14
    config.sample.guidance_scale = 1.0
    config.sample.noise_level = 0.3
    config.train.cfg = False
    config.train.clip_range = 0.2
    config.train.beta = 0.0012
    config.train.lr_scheduler = "step_halving"
    config.train.lr_halving_steps = (25, 30, 160)
    config.train.ema_beta = 0
    config.train.kl_ema_decay = 0.3
    config.train.kl_ema_decay_type = "linear"
    config.train.kl_ema_weight = "Uniform"

    config.value_gradient_estimator = "kde"
    config.kde_bandwidth = 16 * 64 * 64 / 1000.0
    config.policy_gradient_weight = "det_t_exp_a"
    config.weight_exp = 1.0
    config.off_policy_weight = "c_1t_exp_a"
    config.off_policy_weight_exp = 2.0
    config.off_policy_const_factor = 0.05
    config.kl_weight_type = None
    config.kl_weight_exp = 1.0
    config.kl_p_factor = 2.0
    config.skip_is = True
    config.fix = True
    config.fixema = False
    return config


# ==============================================================================
# 6. Qwen-Image + HPSv3 Experiments (Sec. 4.3, Fig. 3(d))
# ==============================================================================

# Paper reference: Sec. 4.3, Fig. 3(d) — Base 64-GPU template for the multi-sample KDE estimator on Qwen-Image HPSv3
def qwenimage_hpsv3_kde_base(gradient_step_per_epoch=2):
    gpu_number = 64
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.pretrained.model = "Qwen/Qwen-Image"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(32 / (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 4  # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.logdir = "logs-qwenimage/"

    if gradient_step_per_epoch == 1:
        config.run_name = "qwenimage_hpsv3_kde_onpolicy"
    else:
        config.run_name = "qwenimage_hpsv3_kde_offpolicy"

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.num_epochs = 250
    config.train.num_inner_epochs = 1
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = False
    config.sample.noise_level = 0.1
    config.reweight_kl = True
    config.start_timesteps = 0
    config.end_timesteps = 9
    config.sample.sde_window_size = 0
    config.sample.sde_window_range = (0, config.sample.num_steps // 2)
    config.mixed_precision = "bf16"
    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.save_freq = 15  # epoch
    config.eval_freq = 15
    config.save_dir = "logs/kde-hps/qwenimage/"
    config.reward_fn = {
        "hpsv3": 1.0,
    }
    config.hpsv3_offload = False

    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


# Paper reference: Sec. 4.3, Fig. 3(d) — Final multi-sample KDE recipe on Qwen-Image HPSv3 (64 GPUs, 1024x1024, off-policy, w2(t)=t^7, w1(t)=(1-t)^3)
def qwenimage_hpsv3_kde():
    config = qwenimage_hpsv3_kde_base(gradient_step_per_epoch=2)
    config.train.cfg = False
    config.run_name += "_1k_w7_offw3_m4000_kl0001"
    config.save_dir += config.run_name
    config.sample.guidance_scale = 1.0
    config.sample.noise_level = 0.1
    config.kde_bandwidth = 64 * 64 * 64 / 4000.0
    config.resolution = 1024

    config.value_gradient_estimator = "kde"
    config.off_policy_weight = "1_1t_exp_a"
    config.policy_gradient_weight = "det_t_exp_a"
    config.weight_exp = 7.0
    config.off_policy_weight_exp = 3.0
    config.train.beta = 0.0001
    return config


# Paper reference: Sec. 4.3, Fig. 3(d) — DiffusionNFT baseline on Qwen-Image HPSv3 (64 GPUs, 1024x1024)
def qwenimage_hpsv3_nft():
    num_prompt_groups = 32
    config = compressibility()

    config.base_model = "qwenimage"
    config.pretrained.model = "Qwen/Qwen-Image"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.prompt_fn = "general_ocr"
    config.reward_fn = {
        "hpsv3": 1.0,
    }
    config.reward_backends = "local"
    config.hpsv3_offload = False

    config.run_name = "qwenimage_hpsv3_nft_64gpu"
    config.logdir = "logs-qwenimage/"
    config.save_dir = "logs/nft-hps/qwenimage/64gpu"
    config.resolution = 1024

    # Keep the Qwen-Image sampling batch layout from pickscore_qwenimage.
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(
        num_prompt_groups
        / (
            64
            * config.sample.train_batch_size
            / config.sample.num_image_per_prompt
        )
    )
    if config.sample.num_batches_per_epoch <= 0:
        raise ValueError("num_batches_per_epoch must be positive")
    config.sample.test_batch_size = 4
    config.sample.global_std = True
    config.sample.same_latent = False

    # Match SD3 NFT sampling and objective settings.
    config.sample.guidance_scale = 1.0
    config.sample.eval_guidance_scale = 1.0
    config.sample.noise_level = 0.7
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    config.sample.sde_window_size = 0
    config.sample.sde_window_range = (0, 9)

    config.train.batch_size = config.sample.train_batch_size
    # One optimizer update consumes all selected prompt groups and timesteps.
    config.train.gradient_accumulation_steps = (
        config.sample.num_batches_per_epoch
    )
    config.num_epochs = 500
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.timestep_selection = "fixed"
    config.start_timesteps = 0
    config.end_timesteps = 9

    config.loss_type = "nft"
    config.decay_type = 1
    config.beta = 1.0
    config.train.beta = 0.0001
    config.train.adv_mode = "all"
    config.train.cfg = False
    config.train.ema = True
    config.train.ema_beta = 0.0
    config.train.kl_ema_weight = "Uniform"
    config.train.kl_ema_decay = 0.3
    config.train.kl_ema_decay_type = "linear"

    config.rationorm = False
    config.mixed_precision = "bf16"
    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.per_prompt_stat_tracking = True
    config.fix = True
    config.eval = True
    config.save = True
    config.eval_freq = 15
    config.save_freq = 15

    return config


# Paper reference: Sec. 4.3, Fig. 3(d) — Flow-GRPO baseline on Qwen-Image HPSv3 (64 GPUs, 1024x1024)
def qwenimage_hpsv3_grpo():
    gpu_number = 64
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.pretrained.model = "Qwen/Qwen-Image"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4

    config.resolution = 1024
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(
        32
        / (
            gpu_number
            * config.sample.train_batch_size
            / config.sample.num_image_per_prompt
        )
    )
    assert config.sample.num_batches_per_epoch % 2 == 0, (
        "Please set config.sample.num_batches_per_epoch to an even number! "
        "This ensures that config.train.gradient_accumulation_steps = "
        "config.sample.num_batches_per_epoch / 2, so that gradients are "
        "updated twice per epoch."
    )
    config.sample.test_batch_size = 4

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = (
        config.sample.num_batches_per_epoch // 2
    )
    config.num_epochs = 250
    config.train.num_inner_epochs = 1
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = False
    config.sample.noise_level = 1.2
    config.sample.sde_window_size = 2
    config.sample.sde_window_range = (0, config.sample.num_steps // 2)
    config.mixed_precision = "bf16"
    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.save_freq = 15
    config.eval_freq = 15
    config.save_dir = "logs/hps/qwenimage"
    config.reward_fn = {
        "hpsv3": 1.0,
    }
    config.hpsv3_offload = False
    config.prompt_fn = "general_ocr"
    config.per_prompt_stat_tracking = True

    config.logdir = "logs-qwenimage/"
    config.run_name = "qwenimage_hpsv3_grpo"
    return config


def get_config(name):
    return globals()[name]()
