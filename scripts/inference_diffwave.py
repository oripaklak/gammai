import datetime
import os
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.diffwave import DiffWave
from utils.helpers import (
    get_noise_schedule,
    load_config,
    normalize_coeffs,
    plot_and_save_spectra,
    plot_diffusion_step,
    save_extracted_peaks_to_files,
    setup_device,
    unnormalize,
    unnormalize_coeffs,
)


def run_reverse_diffusion_loop(
    model,
    config,
    num_samples,
    device,
    track_dir=None,
    global_step=None,
    max_val=None,
    conditioner=None,
    sample_real_coeffs=False,
    bounds_path=None,
):
    """Reverse diffusion engine supporting both 'concat' and 'film' calibration modes.

    Returns:
        tuple: (signal, returned_conditioner)
            - concat mode: signal shape (B, L+4), returned_conditioner shape (B, 4)
            - film mode:   signal shape (B, L),   returned_conditioner shape (B, 4)
    """
    gen_cfg = config["generation"]
    mod_cfg = config["model"]
    calib_cfg = config.get("calibration", {})

    is_calib = calib_cfg.get("enabled", False)
    cond_method = calib_cfg.get("conditioning_method", "concat")
    num_params = calib_cfg.get("num_params", 4)
    fast_sampling = gen_cfg.get("fast_sampling", True)
    full_len = mod_cfg["signal_len"]
    num_tracking_steps = gen_cfg.get("num_tracking_steps", 0)

    # Resolve dataset directory and global calibration bounds path upfront
    base_dir = config["paths"]["base_dir"]
    dataset_name = config["paths"]["dataset_name"]
    dataset_dir = config["paths"].get(
        "dataset_dir", os.path.join(base_dir, dataset_name)
    )
    if bounds_path is None:
        bounds_path = os.path.join(dataset_dir, "global_calib_bounds.json")

        # FiLM conditioning vector generation / sampling logic
    if is_calib and cond_method == "film" and conditioner is None:
        if sample_real_coeffs:
            coeff_path = os.path.join(dataset_dir, "coeff_matrix.npy")

            if not os.path.exists(coeff_path):
                raise FileNotFoundError(
                    f"FiLM calibration enabled but missing coeff matrix at: {coeff_path}"
                )

            raw_coeffs = np.load(coeff_path).astype(np.float32)
            indices = np.random.choice(
                len(raw_coeffs),
                size=num_samples,
                replace=(num_samples > len(raw_coeffs)),
            )
            sampled_raw = raw_coeffs[indices]

            try:
                norm_coeffs = normalize_coeffs(
                    sampled_raw, bounds_path=str(bounds_path)
                )
            except TypeError:
                norm_coeffs = normalize_coeffs(sampled_raw)

            conditioner = torch.from_numpy(norm_coeffs).float().to(device)
        else:
            # Generate valid normalized synthetic calibration vectors strictly in [0, 1]
            conditioner = torch.rand(num_samples, num_params, device=device)

    with torch.no_grad():
        training_noise_schedule, talpha_cum = get_noise_schedule(mod_cfg)
        T = mod_cfg["T"]

        inference_noise_schedule = (
            np.array(mod_cfg["inference_noise_schedule"])
            if fast_sampling
            else training_noise_schedule
        )

        beta = inference_noise_schedule
        alpha = 1.0 - beta
        alpha_cum = np.cumprod(alpha)

        T_steps = []
        for s in range(len(inference_noise_schedule)):
            step_found = False
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                        talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5
                    )
                    T_steps.append(t + twiddle)
                    step_found = True
                    break

            if not step_found:
                if alpha_cum[s] >= talpha_cum[0]:
                    T_steps.append(0.0)
                else:
                    T_steps.append(float(T - 1))

        T_steps = np.array(T_steps, dtype=np.float32)

        # Initialize x_T Gaussian noise (Length L+4 for Concat, L for FiLM)
        signal = torch.randn(num_samples, full_len, device=device)
        noise_scale = torch.from_numpy(alpha_cum**0.5).float().to(device)

        if track_dir and num_tracking_steps > 0:
            last_10_steps = list(range(10))
            extra_5_steps = [199, 150, 100, 50, 20]
            track_indices = set(last_10_steps + extra_5_steps)
            current_track_path = (
                track_dir
                if global_step == "inference"
                else os.path.join(track_dir, f"step_{global_step}")
            )
            os.makedirs(current_track_path, exist_ok=True)
        else:
            track_indices = set()

        # Reverse diffusion loop
        for n in tqdm(range(len(alpha) - 1, -1, -1), desc="Denoising Progress"):
            step_conditioner = (
                signal[:, -4:]
                if (is_calib and cond_method == "concat")
                else conditioner
            )

            if n == len(alpha) - 1 and n in track_indices:
                signal_part = (
                    signal[0, :-4]
                    if (is_calib and cond_method == "concat")
                    else signal[0]
                )
                calib_part = (
                    signal[0, -4:]
                    if (is_calib and cond_method == "concat")
                    else (
                        step_conditioner[0]
                        if step_conditioner is not None
                        else None
                    )
                )
                plot_diffusion_step(
                    current_track_path,
                    signal_part,
                    noise_scale[n].item(),
                    n,
                    "T",
                    max_val,
                    calib_params=calib_part,
                    bounds_path=bounds_path,
                )

            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5

            t_tensor = torch.full((num_samples,), T_steps[n], device=device)
            predicted_noise = model(
                signal, t_tensor, conditioner=step_conditioner
            ).squeeze(1)
            signal = c1 * (signal - c2 * predicted_noise)

            if n > 0:
                noise = torch.randn_like(signal)
                sigma = (
                    (1.0 - alpha_cum[n - 1])
                    / (1.0 - alpha_cum[n])
                    * beta[n]
                ) ** 0.5
                signal += sigma * noise

            if n in track_indices and n != len(alpha) - 1:
                label = f"{n}"
                signal_part = (
                    signal[0, :-4]
                    if (is_calib and cond_method == "concat")
                    else signal[0]
                )
                calib_part = (
                    signal[0, -4:]
                    if (is_calib and cond_method == "concat")
                    else (
                        step_conditioner[0]
                        if step_conditioner is not None
                        else None
                    )
                )
                plot_diffusion_step(
                    current_track_path,
                    signal_part,
                    noise_scale[n].item(),
                    n,
                    label,
                    max_val,
                    calib_params=calib_part,
                    bounds_path=bounds_path,
                )

        signal = torch.clamp(signal, 0.0, 1.0)

    # Determine returned conditioner vector for downstream un-normalization
    if is_calib:
        if cond_method == "concat":
            returned_conditioner = signal[:, -calib_cfg.get("num_params", 4) :]
        else:  # film
            returned_conditioner = conditioner
    else:
        returned_conditioner = None

    return signal, returned_conditioner


def load_model_and_generate(
    config,
    device=torch.device("cuda"),
    track_dir=None,
    max_val=None,
    conditioner=None,
    sample_real_coeffs=False,
    bounds_path=None,
):
    """Initializes model, loads checkpoint weights, and triggers reverse diffusion."""
    weights_path = config["generation"]["model_path"]
    num_samples = config["generation"]["num_samples_to_generate"]

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = DiffWave(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return run_reverse_diffusion_loop(
        model,
        config,
        num_samples,
        device,
        track_dir=track_dir,
        global_step="inference",
        max_val=max_val,
        conditioner=conditioner,
        sample_real_coeffs=sample_real_coeffs,
        bounds_path=bounds_path,
    )


def execute_inference_pipeline(
    config,
    device,
    scaling_path,
    weights_path,
    is_calib,
    num_calib_params,
    cond_method,
    target_conditioner=None,
    bounds_path=None,
    sample_real_coeffs=False,
):
    """Handles IO, execution, slicing, and un-normalization of generated samples."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = os.path.splitext(os.path.basename(weights_path))[0]

    gen_cfg = config["generation"]
    fast_sampling = gen_cfg.get("fast_sampling", True)

    if fast_sampling:
        inference_schedule = config["model"].get(
            "inference_noise_schedule", []
        )
        num_fast_steps = len(inference_schedule)
        model_name = f"{model_name}_T={num_fast_steps}"

    run_name = os.path.basename(os.path.dirname(weights_path))
    if (sample_real_coeffs):
        save_dir = (f"results/{run_name}/{model_name}_{timestamp}")
    else:
        save_dir = f"results/{run_name}/{model_name}_{timestamp}_generated_coeffs"
    os.makedirs(save_dir, exist_ok=True)

    track_dir = os.path.join(save_dir, "denoising_process")
    os.makedirs(track_dir, exist_ok=True)

    max_val = np.load(scaling_path)

    # Run reverse diffusion loop
    signal, returned_conditioner = load_model_and_generate(
        config=config,
        device=device,
        track_dir=track_dir,
        max_val=max_val,
        conditioner=target_conditioner,
        sample_real_coeffs=sample_real_coeffs,
        bounds_path=bounds_path,
    )

    generated_scaled = signal.cpu().numpy()

    # --- Slicing and Un-normalizing Logic ---
    if is_calib:
        if cond_method == "concat":
            gen_spectrum_norm = generated_scaled[:, :-num_calib_params]
            gen_coeffs_norm = generated_scaled[:, -num_calib_params:]

            generated_counts = unnormalize(gen_spectrum_norm, max_val)
            generated_counts = np.clip(
                np.round(generated_counts), a_min=0, a_max=None
            )
            try:
                gen_coeffs = unnormalize_coeffs(
                    gen_coeffs_norm, bounds_path=str(bounds_path)
                )
            except TypeError:
                gen_coeffs = unnormalize_coeffs(gen_coeffs_norm)

        elif cond_method == "film":
            gen_spectrum_norm = generated_scaled  # Signal is purely L length
            generated_counts = unnormalize(gen_spectrum_norm, max_val)
            generated_counts = np.clip(
                np.round(generated_counts), a_min=0, a_max=None
            )

            if returned_conditioner is not None:
                gen_coeffs_norm = returned_conditioner.cpu().numpy()
                try:
                    gen_coeffs = unnormalize_coeffs(
                        gen_coeffs_norm, bounds_path=str(bounds_path)
                    )
                except TypeError:
                    gen_coeffs = unnormalize_coeffs(gen_coeffs_norm)
            else:
                gen_coeffs = None
    else:
        generated_counts = unnormalize(generated_scaled, max_val)
        generated_counts = np.clip(
            np.round(generated_counts), a_min=0, a_max=None
        )
        gen_coeffs = None

    return generated_counts, gen_coeffs, save_dir, model_name


def main(args):
    config = load_config(args.exp, args.model)
    calib_cfg = config.get("calibration", {})
    gen_cfg = config.get("generation", {})

    # CLI parameter overrides
    if args.num_samples is not None:
        config["generation"]["num_samples_to_generate"] = args.num_samples
    if args.model_path is not None:
        config["generation"]["model_path"] = args.model_path

    min_prominence = gen_cfg.get("min_prominence", 10.0)
    detect_peaks = gen_cfg.get("detect_peaks", False)
    is_calib = calib_cfg.get("enabled", False)
    num_calib_params = calib_cfg.get("num_params", 4)
    cond_method = calib_cfg.get("conditioning_method", "concat")

    if is_calib and cond_method == "concat":
        config["model"]["signal_len"] += num_calib_params

    device = setup_device(config["hardware"], mode="inference")
    weights_path = config["generation"]["model_path"]
    num_samples = config["generation"]["num_samples_to_generate"]

    base_dir = config["paths"]["base_dir"]
    dataset_name = config["paths"]["dataset_name"]
    dataset_dir = os.path.join(base_dir, dataset_name)

    config["paths"]["dataset_dir"] = dataset_dir
    scaling_path = os.path.join(dataset_dir, "scaling_factor.npy")
    bounds_path = os.path.join(dataset_dir, "global_calib_bounds.json")

    print(f"Generating {num_samples} samples from {weights_path}...")

    generated_counts, gen_coeffs, save_dir, model_name = (
        execute_inference_pipeline(
            config,
            device,
            scaling_path,
            weights_path,
            is_calib,
            num_calib_params,
            cond_method,
            bounds_path=bounds_path,
            sample_real_coeffs=args.sample_real_coeffs,
        )
    )

    if is_calib and gen_coeffs is not None:
        plot_and_save_spectra(
            generated_counts,
            save_dir,
            model_name,
            gen_coeffs,
            detect_peaks=detect_peaks,
            min_prominence=min_prominence,
        )
        gen_coeff_path = os.path.join(
            save_dir, f"all_{num_samples}_coeffs.npy"
        )
        np.save(gen_coeff_path, gen_coeffs)
    else:
        plot_and_save_spectra(
            generated_counts,
            save_dir,
            model_name,
            detect_peaks=detect_peaks,
            min_prominence=min_prominence,
        )

    # Save peak reports if peak detection is enabled
    if detect_peaks:
        save_extracted_peaks_to_files(
            generated_counts,
            save_dir,
            gen_coeffs if is_calib else None,
            min_prominence=min_prominence,
        )

    data_path = os.path.join(save_dir, f"all_{num_samples}_samples.npy")
    np.save(data_path, generated_counts)
    print(f"Successfully saved all {num_samples} samples to: {save_dir}/")


if __name__ == "__main__":
    parser = ArgumentParser(description="Runs inference for 1-D DiffWave")
    parser.add_argument(
        "--exp",
        type=str,
        default="configs/experiments/diffwave_base.yaml",
        help="Path to experiment config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="configs/models/diffwave.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--sample_real_coeffs",
        action="store_true",
        help="Sample real calibration coefficients from dataset and normalize them for FiLM conditioning",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Override number of samples to generate",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Override model checkpoint weights path",
    )
    main(parser.parse_args())