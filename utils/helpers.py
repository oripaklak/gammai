import yaml
import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt


def load_config(exp_path, model_path):
    """Loads and merges the experiment and model YAML configuration files."""
    with open(exp_path, "r") as file:
        config = yaml.safe_load(file)

    with open(model_path, "r") as file:
        model_config = yaml.safe_load(file)

    # Merge the model config dictionary into the main config dictionary
    config.update(model_config)

    return config


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


def plot_and_save_spectra(generated_counts, save_dir, model_name):
    """Plots and saves a batch of generated spectra to the specified directory."""
    num_samples = generated_counts.shape[0]
    L = generated_counts.shape[1]

    print(f"Saving {num_samples} plots to {save_dir}/ ...")

    for i in range(num_samples):
        counts = generated_counts[i]

        plt.figure(figsize=(10, 5))
        plt.plot(counts, color='blue', linewidth=1)

        # --- Find the last non-zero channel ---
        non_zero_indices = np.nonzero(counts)[0]
        if len(non_zero_indices) > 0:
            max_x = non_zero_indices[-1]
            # Crop the X-axis up to the last value + 50 channels for aesthetic padding
            plt.xlim(0, max_x + 50)
        else:
            # Edge case fallback if the entire array is zeros
            plt.xlim(0, L)

        plt.yscale('log')
        plt.title(f"Generated Spectrum {i + 1} of {num_samples}\nModel: {model_name}")
        plt.xlabel("Channel")
        plt.ylabel("Counts (Log Scale)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()

        # Save each plot
        plot_path = os.path.join(save_dir, f"spectrum_{i + 1:03d}.png")
        plt.savefig(plot_path, dpi=300)

        plt.close()

def normalize(signal, max_val):
    """
    Normalizes the raw signal by applying log1p and dividing by the global maximum.
    Works with both NumPy arrays and PyTorch tensors.
    """
    if isinstance(signal, torch.Tensor):
        return torch.log1p(signal) / max_val
    else:
        return np.log1p(signal) / max_val


def unnormalize(signal, max_val):
    """
    Reverses the normalization to return the signal to its raw integer counts.
    Multiplies by max_val and applies expm1.
    """
    if isinstance(signal, torch.Tensor):
        return torch.expm1(signal * max_val)
    else:
        return np.expm1(signal * max_val)