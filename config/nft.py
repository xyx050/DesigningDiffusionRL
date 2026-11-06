import importlib.util
import os


def _load_base():
    spec = importlib.util.spec_from_file_location("base", os.path.join(os.path.dirname(__file__), "base.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()


def get_config(name):
    return globals()[name]()


def _get_config(base_model="sd3", n_gpus=1, gradient_step_per_epoch=1, dataset="pickscore", reward_fn={}, name=""):
    config = base.get_config()
    assert base_model in ["sd3"]
    assert dataset in ["pickscore", "ocr", "geneval"]

    config.base_model = base_model
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    if base_model == "sd3":
        config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
        config.sample.num_steps = 10
        config.sample.eval_num_steps = 40
        config.sample.guidance_scale = 4.5
        config.resolution = 512
        config.train.beta = 0.0001
        config.sample.noise_level = 0.7
        bsz = 9

    config.sample.num_image_per_prompt = 24
    num_groups = 48

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                config.num_epochs = 700 // gradient_step_per_epoch
                break
        bsz -= 1

    # special design: the test set has a total of 1018/2212/2048 for ocr/geneval/pickscore,
    # to make gpu_num*bs*n as close as possible to it, because when the number of samples
    # cannot be divided evenly by the number of cards, multi-card will fill the last batch
    # to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.test_batch_size = 14 if dataset == "geneval" else 16
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "geneval" if dataset == "geneval" else "general_ocr"

    config.run_name = f"nft_{base_model}_{name}"
    config.save_dir = f"logs/nft/{base_model}/{name}"
    config.reward_fn = reward_fn

    # NFT-specific parameters
    config.loss_type = "nft"       # Loss type: "nft" or "awm"
    config.decay_type = 1          # Decay schedule type for old-adapter update
    config.beta = 1.0              # Blend weight for positive/negative prediction in NFT loss
    config.train.adv_mode = "all"  # Advantage mode: "all", "positive_only", "negative_only", "one_only", "binary"
    config.train.ema_beta = 0.0
    config.train.kl_ema_weight = "Uniform"
    config.train.kl_ema_decay = 0.3
    config.train.kl_ema_decay_type = "linear"

    # Deterministic flow matching sampling settings
    config.sample.guidance_scale = 1.0   # No CFG: deterministic
    config.sample.deterministic = True
    config.sample.solver = "dpm2"        # Flow matching solver for training sampling
    config.train.timestep_fraction = 0.99
    config.train.ema = True
    config.eval_freq = 15
    config.save_freq = 15
    return config

def apply_on_policy(config):
    config.decay_type = 0

    config.run_name += "_onpolicy_woemakl"
    return config

def sd3_ocr():
    reward_fn = {
        "ocr": 1.0,
    }
    config = _get_config(
        base_model="sd3", n_gpus=8, gradient_step_per_epoch=2, dataset="ocr", reward_fn=reward_fn, name="ocr"
    )
    config.beta = 0.1
    config.decay_type = 2
    return config

def sd3_geneval():
    reward_fn = {
        "geneval": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn=reward_fn,
        name="geneval",
    )

    config.eval = True
    config.save = True
    config.reward_backends = "local"
    config.eval_freq = 15

    return config

def sd3_pickscore():
    reward_fn = {
        "pickscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="pickscore",
    )
    config.run_name += "_onpolicy"
    return config

def sd3_pickscore_off_policy():
    reward_fn = {
        "pickscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=2,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="pickscore",
    )
    config.run_name += "_offpolicy"
    return config

def sd3_pickscore_on_policy():
    config = sd3_pickscore()
    config = apply_on_policy(config)
    return config