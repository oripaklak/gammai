import yaml
import torch
import random
import numpy as np


def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def setup_device(hardware_cfg):
    """Configures the torch device based on configuration."""
    num_gpus = hardware_cfg.get("num_gpus", 0)
    device_id = hardware_cfg.get("device_id", 0)

    if num_gpus > 0 and torch.cuda.is_available():
        device_str = f"cuda:{device_id}"
        if num_gpus > 1 and torch.cuda.device_count() > 1:
            print(f"Using {num_gpus} GPUs for training (DataParallel available).")
        else:
            print(f"Training on single GPU: {device_str}")
        return torch.device(device_str)
    else:
        print("Training on CPU...")
        return torch.device("cpu")


def set_seed(seed):
    """Sets the seed for reproducibility across all random number generators."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)