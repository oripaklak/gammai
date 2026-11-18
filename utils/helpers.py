import csv
import datetime
import json
import os
import random
import signal
import sys
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import io
from pathlib import Path 
from typing import Union

# Add photopeaks_automated_identification to sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.abspath(
    os.path.join(_CURRENT_DIR, "..", "photopeaks_automated_identification")
)
if _MODULE_DIR not in sys.path:
  sys.path.insert(0, _MODULE_DIR)

import config
import main

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

    # 1. Model Type (Standard DiffWave vs. Latent Diffusion via VAE)
    vae_enabled = vae_cfg.get("enabled", False)
    if vae_enabled:
        latent_dim = vae_cfg.get("latent_dim", 2048)
        vae_type = str(vae_cfg.get("type", "")).lower()

        # Identify Hugging Face / KVAE-Audio vs. Local VAE
        if vae_type in ["hf", "kvae", "diffusers"] or "repo_id" in vae_cfg:
            repo_id = vae_cfg.get("repo_id", "kandinskylab/KVAE-Audio")
            vae_name = repo_id.split("/")[-1].lower().replace("_", "-")
            model_type = f"stable_diff_{vae_name}_ld{latent_dim}"
        else:
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

def gaussian(x, a, mu, sigma):
    """Standard Gaussian peak function."""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

def load_isotope_dict(nudat_path):
    """Loads nudat_gamma.json."""
    with open(nudat_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def _parse_peaks_csv(
    peak_list: Union[str, Path, "pd.DataFrame"],
) -> dict[int, list[float]]:
    """Parses peak energies grouped by Sample_ID from a CSV path, CSV string, or DataFrame."""
    # Support pandas DataFrame if passed
    if hasattr(peak_list, "iterrows"):
        grouped: dict[int, list[float]] = {}
        for _, row in peak_list.iterrows():
            sample_id = int(row["Sample_ID"]) if "Sample_ID" in row else 1
            grouped.setdefault(sample_id, []).append(float(row["Energy_keV"]))
        return grouped

    # Distinguish between a file path and raw CSV text
    if isinstance(peak_list, Path) or (
        isinstance(peak_list, str) and "\n" not in peak_list.strip()
    ):
        path = Path(peak_list)
        if not path.exists():
            raise FileNotFoundError(f"Peak CSV file not found: {path}")
        csv_text = path.read_text(encoding="utf-8")
    else:
        csv_text = str(peak_list)

    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames or "Energy_keV" not in reader.fieldnames:
        raise ValueError(
            f"Expected 'Energy_keV' column in peak CSV, got columns: {reader.fieldnames}"
        )

    grouped = {}
    for row in reader:
        if not row or not row.get("Energy_keV"):
            continue
        sample_id = int(row["Sample_ID"]) if row.get("Sample_ID") else 1
        grouped.setdefault(sample_id, []).append(float(row["Energy_keV"]))

    return grouped


def _match_sample_isotopes(
    energies: list[float],
    nudat_db: dict[str, list[dict]],
    intensity_threshold: float,
    energy_tol: float,
    min_energy: float,
    require_all: bool,
) -> dict[str, list[float]]:
    """Matches a single spectrum's peak energies (keV) against NuDat gamma lines."""
    valid_peaks = [e for e in energies if e >= min_energy]
    matched_isotopes: dict[str, list[float]] = {}
    assigned_peaks: set[float] = set()

    for isotope, gamma_lines in nudat_db.items():
        if not gamma_lines:
            continue

        # Safely filter reference lines, skipping entries with null energy/intensity
        ref_energies: list[float] = []
        for line in gamma_lines:
            raw_energy = line.get("energy")
            raw_intensity = line.get("intensity")
            if raw_energy is None or raw_intensity is None:
                continue

            energy_val = float(raw_energy)
            intensity_val = float(raw_intensity)
            if intensity_val >= intensity_threshold and energy_val >= min_energy:
                ref_energies.append(energy_val)

        if not ref_energies:
            continue

        # Find observed peaks within +-energy_tol for each qualifying reference line
        line_matches: list[list[float]] = []
        for ref_e in ref_energies:
            hits = [obs_e for obs_e in valid_peaks if abs(obs_e - ref_e) <= energy_tol]
            line_matches.append(hits)

        is_match = (
            all(len(hits) > 0 for hits in line_matches)
            if require_all
            else any(len(hits) > 0 for hits in line_matches)
        )

        if is_match:
            iso_energies = sorted({e for hits in line_matches for e in hits})
            matched_isotopes[isotope] = iso_energies
            assigned_peaks.update(iso_energies)

    unmatched = [e for e in energies if e not in assigned_peaks]
    matched_isotopes["doesn't belong to any isotopes"] = unmatched
    return matched_isotopes


def extract_isotopes(
    peak_list: Union[str, Path, "pd.DataFrame"],
    nudat_path: Union[str, Path, dict],
    intensity_threshold: float = 10.0,
    energy_tol: float = 1.5,
    min_energy: float = 40.0,
    require_all: bool = True,
) -> dict:
    """
    Identifies isotopes from extracted gamma photopeaks using a NuDat JSON database.

    Args:
        peak_list: Path to the extracted peaks CSV file, raw CSV string, or DataFrame
                   with columns: Sample_ID, Peak_Num, Channel, Energy_keV, Counts, Prominence.
        nudat_path: Path to nudat_gamma.json or an already-loaded NuDat dictionary.
        intensity_threshold: Minimum NuDat gamma intensity (%) required for candidate lines.
        energy_tol: Energy matching tolerance in keV (|E_observed - E_nudat| <= energy_tol).
        min_energy: Minimum gamma energy (keV) considered during matching.
        require_all: If True, every NuDat line above intensity_threshold (and >= min_energy)
                     must be detected; if False, at least one qualifying line suffices.

    Returns:
        For a single-sample CSV:
            {<isotope>: [<matched_energies_keV>], "doesn't belong to any isotopes": [<unmatched_keV>]}
        For a multi-sample CSV:
            {<sample_id>: {<isotope>: [...], "doesn't belong to any isotopes": [...]}}
    """
    nudat_db = (
        nudat_path
        if isinstance(nudat_path, dict)
        else load_isotope_dict(nudat_path)
    )
    peaks_by_sample = _parse_peaks_csv(peak_list)

    results_by_sample = {
        sample_id: _match_sample_isotopes(
            energies=energies,
            nudat_db=nudat_db,
            intensity_threshold=intensity_threshold,
            energy_tol=energy_tol,
            min_energy=min_energy,
            require_all=require_all,
        )
        for sample_id, energies in peaks_by_sample.items()
    }

    if len(results_by_sample) == 1:
        return next(iter(results_by_sample.values()))
    return results_by_sample

# def extract_gamma_peaks(
#     spectrum,
#     calib_coeffs=None,
#     fwhm=14,
#     confidence_factor=3.0,
#     category="All",
#     lld=0,
#     hld=None,
#     bounds_path=None,
#     ):
#     """Extracts photopeaks using the Fearn et al. (2022) pipeline."""
#     spectrum = np.asarray(spectrum, dtype=float)
#     L = len(spectrum)
#     channels = np.arange(L)
#     
#     # Calibration
#     if calib_coeffs is not None:
#         calib_np = np.array(calib_coeffs)
#         try:
#           coeffs = unnormalize_coeffs(calib_np, bounds_path=bounds_path)
#         except Exception:
#           coeffs = calib_np
#     if len(coeffs) >= 2:
#       config.c = float(coeffs[0])
#       config.M = float(coeffs[1])
#     else:
#       config.c = 0.0
#       config.M = 1.0
#     else:
#     config.c = 0.0
#     config.M = 1.0
#     
#     # Update config
#     config.FWHM = int(fwhm)
#     config.confidence = float(confidence_factor)
#     config.category = category
#     config.LLD = int(lld)
#     config.HLD = int(hld) if hld is not None else L
#     
#     # Run pipeline
#     df_peaks, fits, FWHMs, net_areas, gross_areas = main.main(spectrum)
#     
#     if df_peaks.empty:
#     return {
#     "indices": np.array([], dtype=int),
#     "energies": np.array([]),
#     "counts": np.array([]),
#     "fwhm": np.array([]),
#     "r2_adjusted": np.array([]),
#     "isotopes": [],
#     "dataframe": df_peaks,
#     "fitted_spectrum": fits,
#     }
#     
#     peak_energies = df_peaks["Energy (keV)"].to_numpy()
#     peak_channels = np.clip(
#     np.round((peak_energies - config.c) / config.M).astype(int), 0, L - 1
#     )
#     peak_counts = spectrum[peak_channels]
#     
#         return {
#         "indices": peak_channels,
#         "energies": peak_energies,
#         "counts": peak_counts,
#         "fwhm": np.asarray(FWHMs),
#         "r2_adjusted": df_peaks[r"$\bar{R}^2$ of fit"].to_numpy(),
#         "isotopes": df_peaks["Isotope"].tolist(),
#         "dataframe": df_peaks,
#         "fitted_spectrum": fits,
#         }

# def extract_gamma_peaks(
#     spectrum,
#     calib_coeffs=None,
#     fwhm=10.0,
#     confidence_factor=3.5,
#     smoothing_iterations=5, 
#     min_distance=20,
#     min_energy=0.0,
#     bounds_path=None,
# ):
#     """Extracts photopeaks using Mariscotti's second-difference algorithm
# 
#     and Gaussian curve fitting based on Fearn et al. (2022).
#     """
#     spectrum = np.asarray(spectrum, dtype=float)
#     L = len(spectrum)
#     channels = np.arange(L)
# 
#     # 1. Energy calibration mapping
#     if calib_coeffs is not None:
#         calib_np = np.array(calib_coeffs)
#         try:
#             coeffs = unnormalize_coeffs(calib_np, bounds_path=bounds_path)
#         except Exception:
#             coeffs = calib_np
#         x_axis = compute_energies(coeffs, L)
#     else:
#         x_axis = channels
# 
#         # 2. Compute smoothed second difference (S_bar) and exact Mariscotti std dev (F_bar)
#         kernel_S = np.array([1.0, -2.0, 1.0])
# 
#         # Compute smoothing window width: w = 2m + 1 ~ 0.65 * FWHM
#         w = max(1, int(round(0.65 * fwhm)))
#         if w % 2 == 0:
#             w += 1
# 
#         # Apply z smoothing iterations via boxcar convolution
#         boxcar = np.ones(w)
#         K = np.ones(1)
#         for _ in range(smoothing_iterations):
#             K = np.convolve(K, boxcar)
# 
#         # Combined second-difference smoothing kernel C = kernel_S * K
#         C = np.convolve(kernel_S, K)
#         S_bar = np.convolve(spectrum, C, mode="same")
#         V_bar = np.convolve(spectrum, C ** 2, mode="same")
#         F_bar = np.sqrt(np.maximum(V_bar, 0.0))
# 
#         # 3. Peak Identification Condition: S_bar < 0, |S_bar| > f * F_bar, and local minimum
#         cond = (S_bar < 0) & (np.abs(S_bar) > confidence_factor * F_bar)
# 
#         # Exclude edge channels distorted by zero-padding in mode="same"
#         pad = max(1, len(C) // 2)
#         local_min = np.zeros(L, dtype=bool)
#         if L > 2 * pad:
#             local_min[pad:-pad] = (
#                     (S_bar[pad:-pad] <= S_bar[pad - 1: -pad - 1])
#                     & (S_bar[pad:-pad] <= S_bar[pad + 1: L - pad + 1])
#             )
# 
#         cand_idx = np.where(cond & local_min)[0]
# 
#     # Ignore lower energies below discriminator threshold if specified
#     if min_energy > 0.0:
#         cand_idx = [idx for idx in cand_idx if x_axis[idx] >= min_energy]
# 
#     # Filter candidate peak indices by minimum distance spacing
#     filtered_idx = []
#     for idx in cand_idx:
#         if not filtered_idx or (idx - filtered_idx[-1]) >= min_distance:
#             filtered_idx.append(idx)
#         else:
#             if S_bar[idx] < S_bar[filtered_idx[-1]]:
#                 filtered_idx[-1] = idx
# 
#     peaks_idx = []
#     peak_energies = []
#     peak_counts = []
#     fit_r2_adjusted = []
#     fit_fwhm = []
# 
#     # 4. Background Subtraction & Gaussian Fitting (Section 2.2)
#     half_fit_win = max(5, int(round(1.5 * fwhm)))
#     bg_n = 10  # 10 channels pre- and post-peak for background averaging
# 
#     for idx in filtered_idx:
#         win_start = max(bg_n, idx - half_fit_win)
#         win_end = min(L - bg_n, idx + half_fit_win + 1)
# 
#         if win_end <= win_start:
#             continue
# 
#         fit_channels = np.arange(win_start, win_end)
#         raw_sub_spec = spectrum[win_start:win_end]
# 
#         # Calculate linear background using 10 channels before and 10 channels after peak
#         bg_pre = np.mean(spectrum[win_start - bg_n : win_start])
#         bg_post = np.mean(spectrum[win_end : win_end + bg_n])
#         bg_line = np.linspace(bg_pre, bg_post, len(fit_channels))
# 
#         bg_subtracted = raw_sub_spec - bg_line
#         bg_subtracted_clean = np.maximum(bg_subtracted, 0.0)
# 
#         # Initial parameter guesses: [Amplitude, Mean, Sigma]
#         amp_guess = np.max(bg_subtracted_clean)
#         mean_guess = float(idx)
#         sigma_guess = max(1.0, fwhm / 2.355)
# 
#         try:
#             popt, _ = curve_fit(
#                 gaussian,
#                 fit_channels,
#                 bg_subtracted_clean,
#                 p0=[amp_guess, mean_guess, sigma_guess],
#                 bounds=(
#                     [0, win_start, 0.1],
#                     [np.inf, win_end, (win_end - win_start)],
#                 ),
#                 maxfev=1000,
#             )
#             fit_amp, fit_mean, fit_sigma = popt
#             fitted_fwhm = 2.355 * abs(fit_sigma)
# 
#             # Compute Adjusted R^2 (Ezekiel's formula)
#             y_pred = gaussian(fit_channels, *popt)
#             ss_res = np.sum((bg_subtracted_clean - y_pred) ** 2)
#             ss_tot = np.sum(
#                 (bg_subtracted_clean - np.mean(bg_subtracted_clean)) ** 2
#             )
# 
#             n = len(fit_channels)
#             p = 3  # parameters: amplitude, mean, sigma
#             if ss_tot > 0 and (n - p - 1) > 0:
#                 r2 = 1.0 - (ss_res / ss_tot)
#                 adj_r2 = 1.0 - (1.0 - r2) * (n - 1.0) / (n - p - 1.0)
#             else:
#                 adj_r2 = 0.0
# 
#             # Interpolate energy at fitted Gaussian centroid
#             fit_energy = (
#                 np.interp(fit_mean, channels, x_axis)
#                 if calib_coeffs is not None
#                 else fit_mean
#             )
# 
#             peaks_idx.append(int(round(fit_mean)))
#             peak_energies.append(fit_energy)
#             peak_counts.append(spectrum[int(round(fit_mean))])
#             fit_r2_adjusted.append(adj_r2)
#             fit_fwhm.append(fitted_fwhm)
#         except Exception:
#             # Fallback to discrete peak channel if non-linear fit fails to converge
#             peaks_idx.append(idx)
#             peak_energies.append(x_axis[idx])
#             peak_counts.append(spectrum[idx])
#             fit_r2_adjusted.append(0.0)
#             fit_fwhm.append(fwhm)
# 
#     return {
#         "indices": np.array(peaks_idx, dtype=int),
#         "energies": np.array(peak_energies),
#         "counts": np.array(peak_counts),
#         "r2_adjusted": np.array(fit_r2_adjusted),
#         "fwhm": np.array(fit_fwhm),
#         "s_bar": S_bar,
#         "f_bar": F_bar,
#     }
# 
def extract_gamma_peaks(
    spectrum,
    calib_coeffs=None,
    min_prominence=10.0,
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

    # Anscombe transform - Stabilize Poisson variance so noise has uniform scale at low and high energy
    spectrum = 2.0 * np.sqrt(np.maximum(spectrum, 0) + 0.375)
    
    smoothed = savgol_filter(spectrum, window_length=window_len, polyorder=3)
    smoothed = np.maximum(smoothed, 0)

    cand_idx, properties = find_peaks(
        smoothed,
        prominence=min_prominence,
        distance=min_distance,
        # width=(2,25), # rejects wide Compton edges while keeping real photopeaks
    )

    refined_idx = []
    # Inverse Anscombe transform 
    spectrum = np.maximum(np.square(0.5 * spectrum) - 0.375, 0.0)
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
    min_distance=50,
    window_len=21,
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
                min_distance=min_distance,
                window_len=window_len,
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

# def save_extracted_peaks_to_files(
#     generated_counts,
#     save_dir,
#     gen_coeffs=None,
#     confidence_factor=3.5,
#     fwhm=10.0,
#     bounds_path=None,
#     min_energy=0.0,
#     min_distance=20,
#     smoothing_iterations=5,
#     **kwargs,
# ):
#   """Extracts photopeaks for all spectra using Mariscotti's second-difference
# 
#   algorithm and Gaussian peak fitting (Fearn et al., 2022), exporting
#   individual peak records to CSV and TXT.
#   """
#   # 1. Ensure output directory exists
#   os.makedirs(save_dir, exist_ok=True)
#   csv_path = os.path.join(save_dir, "extracted_peaks.csv")
#   txt_path = os.path.join(save_dir, "extracted_peaks.txt")
# 
#   generated_counts = np.asarray(generated_counts)
#   num_samples = len(generated_counts)
#   all_rows = []
# 
#   # Determine if gen_coeffs is per-sample (2D/list of lists) or shared (1D)
#   is_per_sample_coeffs = False
#   if gen_coeffs is not None:
#     gen_coeffs_np = np.asarray(gen_coeffs)
#     if gen_coeffs_np.ndim > 1 and len(gen_coeffs_np) == num_samples:
#       is_per_sample_coeffs = True
# 
#   # 2. Process spectra and write human-readable TXT report
#   with open(txt_path, "w", encoding="utf-8") as txt_file:
#     txt_file.write(
#         "=== Extracted Gamma Photopeaks Report ===\n"
#         f"Confidence Factor (f): {confidence_factor}\n"
#         f"Estimated FWHM (channels): {fwhm}\n"
#         f"Total Samples Analyzed: {num_samples}\n"
#         f"Energy Discriminator (min_energy): {min_energy} keV\n\n"
#     )
# 
#     for i in range(num_samples):
#       spectrum = generated_counts[i]
# 
#       # Resolve calibration coefficients for this sample
#       if gen_coeffs is not None:
#         coeffs = gen_coeffs[i] if is_per_sample_coeffs else gen_coeffs
#       else:
#         coeffs = None
# 
#       # Call the Mariscotti + Gaussian fitting algorithm
#       peaks_data = extract_gamma_peaks(
#           spectrum,
#           calib_coeffs=None,
#           fwhm=10.0,
#           confidence_factor=3.5,
#           smoothing_iterations=5,
#           min_distance=20,
#           min_energy=0.0,
#           bounds_path=None,
#       )
# 
#       indices = peaks_data["indices"]
#       energies = peaks_data["energies"]
#       counts = peaks_data["counts"]
#       r2_adj = peaks_data["r2_adjusted"]
#       fwhm_vals = peaks_data["fwhm"]
#       isotopes = peaks_data.get("isotopes", [None] * len(indices))
# 
#       txt_file.write(f"--- Sample {i + 1}/{num_samples} ---\n")
# 
#       if len(indices) == 0:
#         txt_file.write("  No photopeaks detected.\n")
#       else:
#         for p_idx, (ch, E, cnt, r2, peak_fwhm, iso) in enumerate(
#             zip(indices, energies, counts, r2_adj, fwhm_vals, isotopes), start=1
#         ):
#           iso_str = f" | Isotope = {iso}" if iso else ""
#           txt_file.write(
#               f"  Peak {p_idx:02d}: Channel = {ch:4d} | Energy = {E:7.2f} keV | "
#               f"Counts = {cnt:6.0f} | Adj R^2 = {r2:6.4f} | FWHM = {peak_fwhm:5.2f}{iso_str}\n"
#           )
# 
#           row_dict = {
#               "Sample_ID": i + 1,
#               "Peak_Num": p_idx,
#               "Channel": int(ch),
#               "Energy_keV": round(float(E), 2),
#               "Counts": int(cnt),
#               "R2_Adjusted": round(float(r2), 4),
#               "FWHM_Channels": round(float(peak_fwhm), 2),
#           }
#           if iso:
#             row_dict["Isotope"] = iso
#           all_rows.append(row_dict)
# 
#       txt_file.write("\n")
# 
#   # 3. Write CSV file (always write headers even if all_rows is empty)
#   fieldnames = [
#       "Sample_ID",
#       "Peak_Num",
#       "Channel",
#       "Energy_keV",
#       "Counts",
#       "R2_Adjusted",
#       "FWHM_Channels",
#   ]
#   if any("Isotope" in r for r in all_rows):
#     fieldnames.append("Isotope")
# 
#   with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     writer.writeheader()
#     if all_rows:
#       writer.writerows(all_rows)
# 
#   print(
#       f"Successfully saved extracted peaks to:\n"
#       f"  +-- CSV: {csv_path}\n"
#       f"  +-- TXT: {txt_path}\n"
#       f"  +-- Total Peaks Found: {len(all_rows)}"
#   )
#   return all_rows


def generate_synthetic_calib_coeffs(num_samples, num_params=4, device="cuda"):
    """Generates synthetic normalized calibration parameters strictly in [0, 1].

    When passed to unnormalize_coeffs, these values are mathematically guaranteed
    to land strictly inside [calib_min, calib_max].
    """
    # Uniform sampling in [0, 1] ensures valid normalized vectors
    return torch.rand(num_samples, num_params, device=device)