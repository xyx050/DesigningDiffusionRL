import hashlib
import json
import os
import random

import numpy as np
import torch


COMPLETE_NAME = ".complete"
META_NAME = "resume_meta.json"
VOLATILE_CONFIG_KEYS = {
    "eval",
    "eval_freq",
    "logdir",
    "logger",
    "num_checkpoint_limit",
    "num_epochs",
    "resume_from",
    "run_name",
    "save",
    "save_dir",
    "save_freq",
}


def torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _canonical_json_value(value):
    if isinstance(value, dict):
        return {
            str(key): _canonical_json_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def config_fingerprint(config):
    data = config.to_dict()
    for key in VOLATILE_CONFIG_KEYS:
        data.pop(key, None)
    payload = json.dumps(
        _canonical_json_value(data),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expected_metadata(config, trainer, world_size):
    return {
        "config_fingerprint": config_fingerprint(config),
        "trainer": trainer,
        "world_size": int(world_size),
    }


def checkpoint_step(path):
    name = os.path.basename(os.path.normpath(path))
    if not name.startswith("checkpoint-"):
        return -1
    try:
        return int(name.split("checkpoint-", 1)[1])
    except ValueError:
        return -1


def load_metadata(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, META_NAME), "r", encoding="utf-8") as file:
        return json.load(file)


def is_compatible_checkpoint(checkpoint_dir, expected):
    if checkpoint_step(checkpoint_dir) < 0:
        return False
    if not os.path.isfile(os.path.join(checkpoint_dir, COMPLETE_NAME)):
        return False
    if not os.path.isfile(os.path.join(checkpoint_dir, META_NAME)):
        return False
    try:
        metadata = load_metadata(checkpoint_dir)
    except (OSError, json.JSONDecodeError):
        return False
    return all(metadata.get(key) == value for key, value in expected.items())


def find_latest_checkpoint(search_root, expected, include_checkpoints_subdir):
    if not search_root:
        return None
    search_root = os.path.normpath(os.path.expanduser(search_root))
    if not os.path.isdir(search_root):
        return None

    if checkpoint_step(search_root) >= 0:
        return search_root if is_compatible_checkpoint(search_root, expected) else None

    parents = [search_root]
    if include_checkpoints_subdir:
        parents.insert(0, os.path.join(search_root, "checkpoints"))

    candidates = []
    for parent in parents:
        if not os.path.isdir(parent):
            continue
        for name in os.listdir(parent):
            path = os.path.join(parent, name)
            if is_compatible_checkpoint(path, expected):
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=checkpoint_step)


def rng_state_dict():
    state = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state_dict(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
