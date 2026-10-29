import csv
import datetime
import json
import os
import random
import signal
import sys
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def register_graceful_shutdown(learner, config):
    """Registers signal handlers to save a checkpoint on Ctrl+C or kill."""

    def graceful_shutdown(sig, frame):
        if learner.is_master:
            print(f"\n\n[Rank 0] Termination signal ({sig}) detected!")
            print(
                f"Saving emergency checkpoint at Step {learner.step} before exiting..."
            )

            # Force a save right now
            learner.save_to_checkpoint()

            if config["tracking"]["wandb"]["enabled"]:
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
        torch.cuda.set_device(device_id)

        device_str = f"cuda:{device_id}"
        if num_gpus > 1 and torch.cuda.device_count() > 1:
            print(f"Using {num_gpus} GPUs for {mode} (DataParallel available).")
        else:
            print(f"Running {mode} on GPU: {device_str}")
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


def generate_run_name(config, total_params=None):
    """Generates a standardized, compact, and collision-safe run name."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    calib_cfg = config.get("calibration", {})
    loss_cfg = config.get("loss", {})
    vae_cfg = config.get("vae", {})
    path_cfg = config.get("paths", {})

    # 1. Model Type (Standard DiffWave vs. Stable Diffusion via VAE)
    vae_enabled = vae_cfg.get("enabled", False)
    if vae_enabled:
        latent_dim = vae_cfg.get("latent_dim", 2048)
        model_type = f"stable_diff_ld{latent_dim}"
    else:
        model_type = "diff"

    # Format parameter count if provided
    param_tag = ""
    if total_params is not None:
        param_str = format_param_count(total_params)
        param_tag = f"_{param_str}"

    # 2. Dataset & Architecture
    dataset = path_cfg.get("dataset_name", "data")
    layers = model_cfg.get("residual_layers", "unk")
    channels = model_cfg.get("residual_channels", "unk")

    # 3. Training Hyperparameters
    bs = train_cfg.get("batch_size", "unk")
    lr = train_cfg.get("learning_rate", 2e-4)
    seed = train_cfg.get("seed", 42)
    steps = train_cfg.get("max_steps", "unk")
    steps_str = (
        f"{steps // 1000}k"
        if isinstance(steps, int)
        else format_param_count(steps)
    )
    lr_str = f"{lr:.0e}".replace("-0", "-") if isinstance(lr, float) else lr

    # 4. Schedule Target
    schedule = model_cfg.get("schedule_type", "linear")
    if schedule == "cosine":
        end = model_cfg.get("cosine_target_end", 0.001)
    else:
        end = model_cfg.get("beta_final", 0.02)
    end_str = f"{end:.3g}" if isinstance(end, (float, int)) else end

    # 5. Calibration Tag
    calib_str = ""
    if calib_cfg.get("enabled", False):
        method = calib_cfg.get("conditioning_method", "concat")
        if method == "film":
            loc = calib_cfg.get("film_location", "after_output_proj")
            loc_short = (
                loc.replace("after_", "post_")
                .replace("before_", "pre_")
                .replace("output_", "")
            )
            calib_str = f"_film-{loc_short}"
        else:
            calib_str = f"_{method}"

    # 6. NDA Tag (Automatically enforced if VAE is enabled)
    nda_str = ""
    use_nda = vae_enabled or loss_cfg.get("use_nda", train_cfg.get("use_nda", False))
    if use_nda:
        lmbda = loss_cfg.get("lambda_nda", train_cfg.get("lambda_nda", 0.01))
        nda_str = f"_nda{lmbda}"

    # 7. Final String Assembly
    run_name = (
        f"{model_type}{param_tag}_{dataset}_L{layers}C{channels}_"
        f"{schedule}{end_str}_"
        f"bs{bs}_lr{lr_str}_steps{steps_str}"
        f"{calib_str}{nda_str}_"
        f"s{seed}"
    )

    return run_name


def get_noise_schedule(model_cfg):
    """Generates the beta noise schedule and cumulative alpha_bar based on
    config."""
    T = model_cfg["T"]
    schedule_type = model_cfg.get("schedule_type", "linear")

    if schedule_type == "linear":
        beta_init = model_cfg["beta_init"]
        beta_final = model_cfg["beta_final"]
        beta = np.linspace(beta_init, beta_final, T, dtype=np.float32)

    elif schedule_type == "cosine":
        s = model_cfg.get("cosine_s", 0.008)
        target_end = model_cfg.get("cosine_target_end", 0.0)

        t = np.arange(T + 1)

        f_t = np.cos(((t / T + s) / (1.0 + s)) * (np.pi / 2.0)) ** 2
        pure_cosine_alpha_bar = f_t / f_t[0]

        alpha_bar_t = target_end + (1.0 - target_end) * pure_cosine_alpha_bar

        beta = 1.0 - (alpha_bar_t[1:] / alpha_bar_t[:-1])
        beta = np.clip(beta, a_min=0.0, a_max=0.999).astype(np.float32)

    else:
        raise ValueError(f"Unsupported schedule_type: {schedule_type}")

    alpha_bar = np.cumprod(1.0 - beta).astype(np.float32)

    return beta, alpha_bar





def normalize(signal, max_val):
    """Normalizes the raw signal by applying log1p and dividing by the global
    maximum."""
    if isinstance(signal, torch.Tensor):
        return torch.log1p(signal) / max_val
    else:
        return np.log1p(signal) / max_val


def unnormalize(signal, max_val):
    """Reverses the normalization to return the signal to its raw integer
    counts."""
    if isinstance(signal, torch.Tensor):
        return torch.expm1(signal * max_val)
    else:
        return np.expm1(signal * max_val)


def normalize_coeffs(coeffs, bounds_path=None):
    """Applies independent Min-Max scaling to the 4 calibration parameters."""
    if bounds_path is None:
        bounds_path = "data/global_calib_bounds.json"

    if not os.path.exists(bounds_path):
        raise FileNotFoundError(
            f"Calibration bounds file not found at: {bounds_path}"
        )

    with open(bounds_path, "r") as f:
        bounds = json.load(f)

    calib_min = np.array(bounds["calib_min"])
    calib_max = np.array(bounds["calib_max"])

    denominator = np.where(
        calib_max - calib_min == 0, 1e-9, calib_max - calib_min
    )
    norm_coeffs = (coeffs - calib_min) / denominator
    return norm_coeffs


def unnormalize_coeffs(norm_coeffs, bounds_path=None):
    """Reverses the Min-Max scaling for the calibration parameters."""
    if bounds_path is None:
        bounds_path = "data/global_calib_bounds.json"

    if not os.path.exists(bounds_path):
        raise FileNotFoundError(
            f"Calibration bounds file not found at: {bounds_path}"
        )

    with open(bounds_path, "r") as f:
        bounds = json.load(f)

    calib_min = np.array(bounds["calib_min"])
    calib_max = np.array(bounds["calib_max"])

    raw_coeffs = norm_coeffs * (calib_max - calib_min) + calib_min
    return raw_coeffs


def compute_energies(coeffs, num_channels):
    """Evaluates polynomial calibration coefficients while enforcing:
    1. Strictly non-negative energy values (E >= 0.0).
    2. Monotonic non-decreasing progression across channels.
    """
    channels = np.arange(num_channels)
    if coeffs is not None and len(coeffs) == 4:
        energies = np.polyval(coeffs[::-1], channels)
    else:
        energies = channels.astype(np.float32)

    energies = np.maximum(energies, 0.0)
    energies = np.maximum.accumulate(energies)
    return energies

def extract_gamma_peaks(
    spectrum,
    calib_coeffs=None,
    min_prominence=25.0,
    min_distance=50,
    window_len=21,
    bounds_path=None,
):
    """Extracts photopeaks with index refinement to ensure the raw maximum count
    is recorded."""
    L = len(spectrum)
    channels = np.arange(L)

    if calib_coeffs is not None:
        calib_np = np.array(calib_coeffs)
        try:
            coeffs = unnormalize_coeffs(calib_np, bounds_path=bounds_path)
        except Exception:
            coeffs = calib_np
        x_axis = compute_energies(coeffs, L)
    else:
        x_axis = channels

    smoothed = savgol_filter(spectrum, window_length=window_len, polyorder=2)
    smoothed = np.maximum(smoothed, 0)

    cand_idx, properties = find_peaks(
        smoothed, prominence=min_prominence, distance=min_distance
    )

    refined_idx = []
    for idx in cand_idx:
        win_start = max(0, idx - 2)
        win_end = min(L, idx + 3)
        local_max_rel = np.argmax(spectrum[win_start:win_end])
        refined_idx.append(win_start + local_max_rel)

    peaks_idx = np.array(refined_idx, dtype=int)

    peak_energies = x_axis[peaks_idx]
    peak_counts = spectrum[peaks_idx]
    prominences = properties["prominences"]

    return {
        "indices": peaks_idx,
        "energies": peak_energies,
        "counts": peak_counts,
        "prominences": prominences,
    }


def save_extracted_peaks_to_files(
    generated_counts,
    save_dir,
    gen_coeffs=None,
    min_prominence=10.0,
    bounds_path=None,
):
    """Extracts photopeaks for all generated spectra and exports to CSV and
    TXT."""
    csv_path = os.path.join(save_dir, "extracted_peaks.csv")
    txt_path = os.path.join(save_dir, "extracted_peaks.txt")

    num_samples = len(generated_counts)
    all_rows = []

    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(
            f"=== Extracted Gamma Photopeaks Report (Prominence >= {min_prominence}, {num_samples} Samples) ===\n\n"
        )

        for i in range(num_samples):
            spectrum = generated_counts[i]
            coeffs = gen_coeffs[i] if gen_coeffs is not None else None

            peaks_data = extract_gamma_peaks(
                spectrum,
                calib_coeffs=coeffs,
                min_prominence=min_prominence,
                bounds_path=bounds_path,
            )

            indices = peaks_data["indices"]
            energies = peaks_data["energies"]
            counts = peaks_data["counts"]
            prominences = peaks_data["prominences"]

            txt_file.write(f"--- Sample {i + 1}/{num_samples} ---\n")
            if len(indices) == 0:
                txt_file.write("  No peaks detected.\n")
            else:
                for p_idx, (ch, E, cnt, prom) in enumerate(
                    zip(indices, energies, counts, prominences), start=1
                ):
                    txt_file.write(
                        f"  Peak {p_idx:02d}: Channel = {ch:4d} | Energy = {E:7.2f} keV | "
                        f"Counts = {cnt:6.0f} | Prominence = {prom:6.2f}\n"
                    )

                    all_rows.append({
                        "Sample_ID": i + 1,
                        "Peak_Num": p_idx,
                        "Channel": ch,
                        "Energy_keV": round(float(E), 2),
                        "Counts": int(cnt),
                        "Prominence": round(float(prom), 2),
                    })
            txt_file.write("\n")

    if all_rows:
        fieldnames = [
            "Sample_ID",
            "Peak_Num",
            "Channel",
            "Energy_keV",
            "Counts",
            "Prominence",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    print(
        f"Successfully saved extracted peaks to:\n  +-- CSV (Excel): {csv_path}\n  +-- TXT:         {txt_path}"
    )


def generate_synthetic_calib_coeffs(num_samples, num_params=4, device="cuda"):
    """Generates synthetic normalized calibration parameters strictly in [0, 1].

    When passed to unnormalize_coeffs, these values are mathematically guaranteed
    to land strictly inside [calib_min, calib_max].
    """
    # Uniform sampling in [0, 1] ensures valid normalized vectors
    return torch.rand(num_samples, num_params, device=device)