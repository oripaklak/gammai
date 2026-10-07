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


def generate_run_name(config, total_params):
    """Generates a standardized, descriptive directory name for a training run
    based on the model and training configurations."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    calib_cfg = config.get("calibration", {})
    path_cfg = config.get("paths", {})

    # Model and training specs
    layers = model_cfg.get("residual_layers", "unknown")
    channels = model_cfg.get("residual_channels", "unknown")
    schedule_type = model_cfg.get("schedule_type", "linear")
    batch_size = train_cfg.get("batch_size", "unknown")
    epochs = train_cfg.get("epochs", "unknown")

    # NDA flag lookup
    use_nda = train_cfg.get("use_nda", False)

    # Calibration specs
    calib_enabled = calib_cfg.get("enabled", False)
    cond_method = calib_cfg.get("conditioning_method", "concat")
    film_location = calib_cfg.get("film_location", "after_output_proj")

    # Dataset name
    dataset_name = path_cfg.get("dataset_name", "data")

    # Extract target end for noise schedule
    if schedule_type == "cosine":
        end = model_cfg.get("cosine_target_end", "unknown")
    elif schedule_type == "linear":
        end = model_cfg.get("beta_final", "unknown")
    else:
        end = "unknown"

    end_str = f"{end:.4g}" if isinstance(end, (float, int)) else end
    param_str = format_param_count(total_params)

    # 1. Base run string
    base_name = f"diff_{param_str}_L{layers}_C{channels}_bs{batch_size}_{schedule_type}-{end_str}_ep{epochs}"

    # 2. Calibration suffix
    calib_suffix = ""
    if calib_enabled:
        if cond_method == "film":
            calib_suffix = f"_calib_{cond_method}_{film_location}"
        else:
            calib_suffix = f"_calib_{cond_method}"

    # 3. NDA suffix (placed before dataset)
    nda_suffix = "_nda" if use_nda else ""

    # Assemble final name
    run_name = f"{base_name}{calib_suffix}{nda_suffix}_{dataset_name}"
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


def plot_noise_schedule(beta, alpha_bar, run_dir):
    """Generates and saves visual plots for the diffusion noise schedules."""
    save_dir = os.path.join(run_dir, "tracked_noise")
    os.makedirs(save_dir, exist_ok=True)

    T = len(beta)
    steps = np.arange(T)

    plt.figure(figsize=(10, 5))
    plt.plot(
        steps, beta, label="Beta (Variance added)", color="red", linewidth=2
    )
    plt.title(f"Beta Noise Schedule (T={T})")
    plt.xlabel("Diffusion Step (t)")
    plt.ylabel("Beta Value")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "beta_schedule.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        steps,
        alpha_bar,
        label="Alpha Bar (Signal Preserved)",
        color="blue",
        linewidth=2,
    )
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


def compute_safe_energies(coeffs, num_channels):
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


def plot_and_save_spectra(
    generated_counts,
    save_dir,
    model_name,
    gen_coeffs=None,
    detect_peaks=True,
    min_prominence=25.0,
    bounds_path=None,
):
    """Plots final generated spectra in Energy (keV) space."""
    os.makedirs(save_dir, exist_ok=True)
    num_samples = len(generated_counts)

    for i in range(num_samples):
        spectrum = generated_counts[i]
        coeffs = gen_coeffs[i] if gen_coeffs is not None else None

        L = len(spectrum)
        channels = np.arange(L)

        fig, ax = plt.subplots(figsize=(12, 4.5))

        if coeffs is not None and len(coeffs) == 4:
            x_axis = compute_safe_energies(coeffs, L)
            x_label = "Energy (keV)"
            max_e = np.max(x_axis) if len(x_axis) > 0 else 3000.0
            xlim_right = float(np.clip(max_e, 30.0, 3000.0))
            ax.set_xlim(0.0, xlim_right)
        else:
            x_axis = channels
            x_label = "Channel"
            ax.set_xlim(0, L)

        sig_plot = np.maximum(spectrum, 1e-1)
        ax.plot(
            x_axis,
            sig_plot,
            color="blue",
            linewidth=1,
            label="Generated Spectrum",
        )

        energies, counts = [], []
        if detect_peaks:
            peaks_data = extract_gamma_peaks(
                spectrum,
                calib_coeffs=coeffs,
                min_prominence=min_prominence,
                min_distance=50,
                bounds_path=bounds_path,
            )

            energies = peaks_data["energies"]
            counts = peaks_data["counts"]

            sorted_order = np.argsort(energies)
            energies = energies[sorted_order]
            counts = counts[sorted_order]

            for idx, (p_x, p_y) in enumerate(zip(energies, counts)):
                if coeffs is not None and (p_x < 0.0 or p_x > 3000.0):
                    continue
                y_val = max(p_y, 1e-1)
                stagger_level = idx % 3
                y_text_mult = 2.2 * (2.2**stagger_level)

                ax.annotate(
                    f"{p_x:.1f}",
                    xy=(p_x, y_val * 1.12),
                    xytext=(p_x, y_val * y_text_mult),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    color="darkred",
                    rotation=90,
                    arrowprops=dict(
                        arrowstyle="->",
                        color="red",
                        lw=0.8,
                        shrinkA=0,
                        shrinkB=4,
                    ),
                )

        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-1)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Counts (Log Scale)")

        peak_count_str = len(energies) if detect_peaks else 0
        ax.set_title(
            f"Generated Spectrum {i + 1}/{num_samples} | Model: {model_name} | Detected Peaks: {peak_count_str}"
        )
        ax.grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"spectrum_{i+1:03d}.png"), dpi=300)
        plt.close(fig)

def plot_diffusion_step(
    track_path,
    single_signal,
    noise_scale_val,
    loop_idx,
    label,
    max_val=None,
    calib_params=None,
    bounds_path=None,
):
    """Saves a PNG graph of a single spectrum at a specific step in the diffusion loop."""
    if isinstance(single_signal, torch.Tensor):
        sig_np = single_signal.detach().cpu().numpy()
    else:
        sig_np = single_signal

    L = len(sig_np)

    if max_val is not None:
        sig_np = unnormalize(sig_np, max_val)
        sig_np = np.clip(np.round(sig_np), a_min=0, a_max=None)
        y_label = "Counts (Log Scale)"
        sig_plot = np.maximum(sig_np, 1e-1)
    else:
        y_label = "Amplitude (Normalized)"
        sig_plot = sig_np

    fig, ax = plt.subplots(figsize=(10, 4))

    if calib_params is not None:
        if isinstance(calib_params, torch.Tensor):
            calib_np = calib_params.detach().cpu().numpy()
        else:
            calib_np = np.array(calib_params)

        if bounds_path is None:
            bounds_path = "data/Baq/global_calib_bounds.json"

        try:
            coeffs = unnormalize_coeffs(calib_np, bounds_path=str(bounds_path))
        except Exception:
            coeffs = calib_np

        # Always evaluate energies in keV when calib_params are present
        x_axis = compute_safe_energies(coeffs, L)
        x_label = "Energy (keV)"
        title_ext = " | Calibrated"

        # Enforce X-axis limits: starts at 0.0, ends between 30.0 and 3000.0 keV
        max_e = np.max(x_axis) if len(x_axis) > 0 else 3000.0
        xlim_right = float(np.clip(max_e, 30.0, 3000.0))
        ax.set_xlim(0.0, xlim_right)
    else:
        x_axis = np.arange(L)
        x_label = "Channel"
        title_ext = ""
        ax.set_xlim(0, L)

    ax.plot(x_axis, sig_plot, color="blue", linewidth=1)

    if max_val is not None:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-1)

    ax.set_title(
        f"Diffusion Generation Progress [{str(label).upper()}]{title_ext} | Loop Index: {loop_idx}\nNoise Scale: {noise_scale_val:.6f}"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(track_path, f"{label}_plot.png"), dpi=300)
    plt.close(fig)

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
        x_axis = compute_safe_energies(coeffs, L)
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


if __name__ == "__main__":
    print("Running standalone noise schedule comparison...")

    base_model_cfg = {"T": 200, "beta_init": 0.0001, "beta_final": 0.02}

    linear_cfg = base_model_cfg.copy()
    linear_cfg["schedule_type"] = "linear"
    beta_lin, alpha_lin = get_noise_schedule(linear_cfg)

    cosine_cfg = base_model_cfg.copy()
    cosine_cfg["schedule_type"] = "cosine"
    cosine_cfg["cosine_s"] = 0.008
    cosine_cfg["cosine_target_end"] = 0.01
    beta_cos, alpha_cos = get_noise_schedule(cosine_cfg)

    save_dir = "noise_schedules_test/comparison"
    os.makedirs(save_dir, exist_ok=True)

    T = len(beta_lin)
    steps = np.arange(T)

    plt.figure(figsize=(10, 5))
    plt.plot(
        steps,
        beta_lin,
        label="Linear Beta",
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(steps, beta_cos, label="Cosine Beta", color="blue", linewidth=2)
    plt.title(f"Beta Noise Schedule Comparison (T={T})")
    plt.xlabel("Diffusion Step (t)")
    plt.ylabel("Beta Value")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "beta_comparison.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        steps,
        alpha_lin,
        label="Linear Alpha Bar",
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        steps,
        alpha_cos,
        label="Cosine Alpha Bar",
        color="blue",
        linewidth=2,
    )
    plt.title(f"Cumulative Signal Preservation Comparison (T={T})")
    plt.xlabel("Diffusion Step (t)")
    plt.ylabel("Alpha Bar Value")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "alpha_cum_comparison.png"), dpi=300)
    plt.close()

    print(f"\nSuccess! Comparison plots saved to: {save_dir}/")