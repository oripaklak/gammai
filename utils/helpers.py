import yaml
import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import signal
import sys
import json

def register_graceful_shutdown(learner, config):
    """Registers signal handlers to save a checkpoint on Ctrl+C or kill."""

    def graceful_shutdown(sig, frame):
        if learner.is_master:
            print(f"\n\n[Rank 0] Termination signal ({sig}) detected!")
            print(f"Saving emergency checkpoint at Step {learner.step} before exiting...")

            # Force a save right now
            learner.save_to_checkpoint()

            if config['tracking']['wandb']['enabled']:
                import wandb
                wandb.finish()

            print("Emergency save complete. Exiting safely.")

        sys.exit(0)

    # Attach the handler to Ctrl+C (SIGINT) and kill commands (SIGTERM)
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

def load_config(exp_path, model_path):
    """Loads and merges the experiment and model YAML configuration files."""
    with open(exp_path, "r") as file:
        config = yaml.safe_load(file)

    with open(model_path, "r") as file:
        model_config = yaml.safe_load(file)

    # Merge the model config dictionary into the main config dictionary
    config.update(model_config)

    return config


def setup_device(hardware_cfg, mode="training"):
    """Configures the torch device based on configuration."""
    num_gpus = hardware_cfg.get("num_gpus", 0)
    device_id = hardware_cfg.get("device_id", 0)

    if num_gpus > 0 and torch.cuda.is_available():
        device_str = f"cuda:{device_id}"
        if num_gpus > 1 and torch.cuda.device_count() > 1:
            # Replaced "training" with the {mode} variable
            print(f"Using {num_gpus} GPUs for {mode} (DataParallel available).")
        else:
            print(f"Running {mode} on single GPU: {device_str}")
        return torch.device(device_str)
    else:
        print(f"Running {mode} on CPU...")
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

def format_param_count(num_params):
    """Formats a parameter count into a readable string like '15k' or '20M'."""
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.0f}M".replace(".0M", "M")
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.0f}k".replace(".0k", "k")
    return f"{num_params}"


def generate_run_name(config, total_params):
    """
    Generates a standardized, descriptive directory name for a training run
    based on the model and training configurations.
    """
    layers = config['model']['residual_layers']
    channels = config['model']['residual_channels']
    schedule_type = config['model']['schedule_type']
    calib = config['experiment']['calibration']

    # Safely extract the schedule's end value
    if schedule_type == "cosine":
        end = config['model'].get('cosine_target_end', "unknown")
    elif schedule_type == "linear":
        end = config['model'].get('beta_final', "unknown")
    else:
        end = "unknown"

    end_str = f"{end:.4g}" if isinstance(end, (float, int)) else end
    param_str = format_param_count(total_params)

    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']

    if not calib:
        run_name = f"diff_{param_str}_L{layers}_C{channels}_bs{batch_size}_{schedule_type}-{end_str}_ep{epochs}_IAEA_Baq"
    else:
        run_name = f"diff_{param_str}_L{layers}_C{channels}_bs{batch_size}_{schedule_type}-{end_str}_ep{epochs}_calib_IAEA_Baq"

    return run_name

def get_noise_schedule(model_cfg):
    """Generates the beta noise schedule and cumulative alpha_bar based on config."""
    T = model_cfg['T']
    schedule_type = model_cfg.get('schedule_type', 'linear')

    if schedule_type == 'linear':
        beta_init = model_cfg['beta_init']
        beta_final = model_cfg['beta_final']
        beta = np.linspace(beta_init, beta_final, T, dtype=np.float32)

    elif schedule_type == 'cosine':
        s = model_cfg.get('cosine_s', 0.008)
        # --- NEW: Fetch the target end directly from the config (default to 0.0) ---
        target_end = model_cfg.get('cosine_target_end', 0.0)

        t = np.arange(T + 1)

        # 1. Calculate the pure cosine curve (goes from 1 to 0)
        f_t = np.cos(((t / T + s) / (1.0 + s)) * (np.pi / 2.0)) ** 2
        pure_cosine_alpha_bar = f_t / f_t[0]

        # 2. Scale the cosine curve so it ends at target_end instead of 0
        alpha_bar_t = target_end + (1.0 - target_end) * pure_cosine_alpha_bar

        # 3. Reverse-engineer the betas from the scaled alpha_bar
        beta = 1.0 - (alpha_bar_t[1:] / alpha_bar_t[:-1])
        beta = np.clip(beta, a_min=0.0, a_max=0.999).astype(np.float32)

    else:
        raise ValueError(f"Unsupported schedule_type: {schedule_type}")

    # Calculate final cumulative noise (alpha_bar) to return
    alpha_bar = np.cumprod(1.0 - beta).astype(np.float32)

    return beta, alpha_bar


def plot_noise_schedule(beta, alpha_bar, run_dir):
    """
    Generates and saves visual plots for the diffusion noise schedules.
    Saves a plot for the Beta schedule and the Alpha Bar (Cumulative) schedule.
    """
    # Create the specific output directory
    save_dir = os.path.join(run_dir, "tracked_noise")
    os.makedirs(save_dir, exist_ok=True)

    T = len(beta)
    steps = np.arange(T)

    # --- Plot 1: Beta Schedule (Noise Variance added per step) ---
    plt.figure(figsize=(10, 5))
    plt.plot(steps, beta, label="Beta (Variance added)", color='red', linewidth=2)
    plt.title(f"Beta Noise Schedule (T={T})")
    plt.xlabel("Diffusion Step (t)")
    plt.ylabel("Beta Value")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "beta_schedule.png"), dpi=300)
    plt.close()

    # --- Plot 2: Alpha Bar (Cumulative Signal Remaining) ---
    plt.figure(figsize=(10, 5))
    plt.plot(steps, alpha_bar, label="Alpha Bar (Signal Preserved)", color='blue', linewidth=2)
    plt.title(f"Cumulative Signal Preservation (T={T})")
    plt.xlabel("Diffusion Step (t)")
    plt.ylabel("Alpha Bar Value")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "alpha_cum_schedule.png"), dpi=300)
    plt.close()

    print(f"Noise schedule plots saved to: {save_dir}/")

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
def normalize_coeffs(coeffs, bounds_path="data/global_calib_bounds.json"):
    """
    Applies independent Min-Max scaling to the 4 calibration parameters.
    Maps values to the [0, 1] range based on global bounds.
    """
    with open(bounds_path, "r") as f:
        bounds = json.load(f)

    calib_min = np.array(bounds["calib_min"])
    calib_max = np.array(bounds["calib_max"])

    # Safe denominator to prevent division by zero
    denominator = np.where(calib_max - calib_min == 0, 1e-9, calib_max - calib_min)

    norm_coeffs = (coeffs - calib_min) / denominator
    return norm_coeffs
def unnormalize_coeffs(norm_coeffs, bounds_path="data/global_calib_bounds.json"):
    """
    Reverses the Min-Max scaling for the calibration parameters,
    returning them to their original orders of magnitude.
    """
    with open(bounds_path, "r") as f:
        bounds = json.load(f)

    calib_min = np.array(bounds["calib_min"])
    calib_max = np.array(bounds["calib_max"])

    raw_coeffs = norm_coeffs * (calib_max - calib_min) + calib_min
    return raw_coeffs


def plot_and_save_spectra(generated_counts, save_dir, model_name, gen_coeffs=None):
    """Plots and saves a batch of generated spectra to the specified directory."""
    num_samples = generated_counts.shape[0]
    L = generated_counts.shape[1]

    print(f"Saving {num_samples} plots to {save_dir}/ ...")

    for i in range(num_samples):
        counts = generated_counts[i]

        plt.figure(figsize=(10, 5))

        # --- Calibration Logic ---
        if gen_coeffs is not None:
            coeffs = gen_coeffs[i]
            channels = np.arange(L)
            x_axis = np.polyval(coeffs[::-1], channels)
            x_label = "Energy (keV)"
            title_ext = "(Calibrated)"
        else:
            x_axis = np.arange(L)
            x_label = "Channel"
            title_ext = "(Uncalibrated)"

        plt.plot(x_axis, counts, color='blue', linewidth=1)

        # --- Find the last non-zero channel/energy ---
        non_zero_indices = np.nonzero(counts)[0]
        if len(non_zero_indices) > 0:
            max_idx = non_zero_indices[-1]
            max_x = x_axis[max_idx]
            pad = 50 if gen_coeffs is None else (x_axis[min(max_idx + 50, L - 1)] - max_x)
            plt.xlim(max(0, x_axis[0]), max_x + pad)
        else:
            plt.xlim(max(0, x_axis[0]), x_axis[-1])

        plt.yscale('log')
        plt.title(f"Generated Spectrum {i + 1} of {num_samples} {title_ext}\nModel: {model_name}")
        plt.xlabel(x_label)
        plt.ylabel("Counts (Log Scale)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()

        plot_path = os.path.join(save_dir, f"spectrum_{i + 1:03d}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

def plot_diffusion_step(track_path, single_signal, noise_scale_val, loop_idx, label, max_val=None):
    """
    MAIN GOAL: The Photographer.
    A purely visual helper that saves a PNG graph of a single spectrum at a specific step in time.
    """
    sig_np = single_signal.cpu().numpy()

    if max_val is not None:
        sig_np = unnormalize(sig_np, max_val)
        sig_np = np.clip(np.round(sig_np), a_min=0, a_max=None)
        y_label = "Counts (Log Scale)"
    else:
        y_label = "Amplitude (Normalized)"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sig_np, color='blue', linewidth=1)

    # Dynamic cropping ONLY for the final clean step
    if label == "0":
        non_zero_indices = np.nonzero(sig_np)[0]
        if len(non_zero_indices) > 0:
            max_x = non_zero_indices[-1]
            ax.set_xlim(0, max_x + 50)
        else:
            ax.set_xlim(0, len(sig_np))


    if max_val is not None:
        ax.set_yscale('log')

    ax.set_title(
        f"Diffusion Generation Progress [{label.upper()}] | Loop Index: {loop_idx}\nNoise Scale: {noise_scale_val:.6f}")
    ax.set_xlabel("Channel")
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(track_path, f"{label}_plot.png"), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    print("Running standalone noise schedule comparison...")

    # Mock configuration based on your diffwave.yaml parameters
    base_model_cfg = {
        'T': 200,
        'beta_init': 0.0001,
        'beta_final': 0.02
    }

    # --- 1. Generate Both Schedules ---
    linear_cfg = base_model_cfg.copy()
    linear_cfg['schedule_type'] = 'linear'
    beta_lin, alpha_lin = get_noise_schedule(linear_cfg)

    cosine_cfg = base_model_cfg.copy()
    cosine_cfg['schedule_type'] = 'cosine'
    cosine_cfg['cosine_s'] = 0.008
    cosine_cfg['cosine_target_end'] = 0.01
    beta_cos, alpha_cos = get_noise_schedule(cosine_cfg)

    # --- 2. Setup Directory ---
    save_dir = "noise_schedules_test/comparison"
    os.makedirs(save_dir, exist_ok=True)

    T = len(beta_lin)
    steps = np.arange(T)

    # --- 3. Plot Beta Comparison ---
    plt.figure(figsize=(10, 5))
    plt.plot(steps, beta_lin, label="Linear Beta", color='red', linestyle='--', linewidth=2)
    plt.plot(steps, beta_cos, label="Cosine Beta", color='blue', linewidth=2)
    plt.title(f"Beta Noise Schedule Comparison (T={T})")
    plt.xlabel("Diffusion Step (t)")
    plt.ylabel("Beta Value")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "beta_comparison.png"), dpi=300)
    plt.close()

    # --- 4. Plot Alpha Cum Comparison ---
    plt.figure(figsize=(10, 5))
    plt.plot(steps, alpha_lin, label="Linear Alpha Bar", color='red', linestyle='--', linewidth=2)
    plt.plot(steps, alpha_cos, label="Cosine Alpha Bar", color='blue', linewidth=2)
    plt.title(f"Cumulative Signal Preservation Comparison (T={T})")
    plt.xlabel("Diffusion Step (t)")
    plt.ylabel("Alpha Bar Value")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "alpha_cum_comparison.png"), dpi=300)
    plt.close()

    print(f"\nSuccess! Comparison plots saved to: {save_dir}/")