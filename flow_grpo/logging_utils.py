import json
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb as _wandb


class ExperimentLogger:
    """Unified logger supporting 'wandb' (default), 'tensorboard', or 'all'."""

    def __init__(self):
        self.use_wandb = True
        self.use_tb = False
        self.writer = None

    @property
    def Image(self):
        return _wandb.Image

    def init(self, project="diffusion-rl", name=None, config=None, dir=None, **kwargs):
        cfg_dict = config if isinstance(config, dict) else {}
        logger_type = str(cfg_dict.get("logger", "wandb")).lower()
        logdir = dir or cfg_dict.get("logdir", "logs")

        self.use_wandb = logger_type in ("wandb", "all")
        self.use_tb = logger_type in ("tensorboard", "all")

        if self.use_wandb:
            init_kwargs = {"project": project, "name": name, "config": config, **kwargs}
            if dir is not None:
                init_kwargs["dir"] = dir
            _wandb.init(**init_kwargs)

        if self.use_tb:
            tb_log_dir = os.path.join(logdir, name or "", "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_log_dir)
            if config is not None:
                self.writer.add_text("config", json.dumps(config, indent=2, default=str), 0)

    def log(self, data, step=None, commit=None, **kwargs):
        if self.use_wandb:
            log_kwargs = dict(kwargs)
            if step is not None:
                log_kwargs["step"] = step
            if commit is not None:
                log_kwargs["commit"] = commit
            _wandb.log(data, **log_kwargs)

        if self.use_tb and self.writer is not None and isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    self.writer.add_scalar(key, float(value), step)
                elif isinstance(value, torch.Tensor) and value.numel() == 1:
                    self.writer.add_scalar(key, float(value.detach().cpu().item()), step)
                elif isinstance(value, np.ndarray) and value.size == 1:
                    self.writer.add_scalar(key, float(value.item()), step)
                elif isinstance(value, list) and value and hasattr(value[0], "image"):
                    for idx, item in enumerate(value):
                        pil_img = getattr(item, "image", None)
                        if pil_img is not None:
                            arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
                            self.writer.add_image(f"{key}/{idx}", arr.transpose(2, 0, 1), step)
                        caption = getattr(item, "_caption", None)
                        if caption:
                            self.writer.add_text(f"{key}_caption/{idx}", str(caption), step)

    def finish(self):
        if self.use_wandb:
            _wandb.finish()
        if self.writer is not None:
            self.writer.close()
            self.writer = None
