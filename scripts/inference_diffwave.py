import numpy as np
import os
import sys
import datetime
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helpers import (load_config, setup_device, plot_and_save_spectra,
                           unnormalize, unnormalize_coeffs,
                           get_noise_schedule, plot_diffusion_step)
from models.diffwave import DiffWave


def run_reverse_diffusion_loop(model, config, num_samples, device, track_dir=None, global_step=None, max_val=None,
                               conditioner=None):
    gen_cfg = config['generation']
    mod_cfg = config['model']
    calib_cfg = config.get('calibration', {})

    is_calib = calib_cfg.get('enabled', False)
    cond_method = calib_cfg.get('conditioning_method', 'concat')
    fast_sampling = gen_cfg.get('fast_sampling', True)
    signal_len = mod_cfg['signal_len']

    # FIX: Determine length based on method
    full_len = signal_len + calib_cfg.get('num_params', 4) if (is_calib and cond_method == 'concat') else signal_len
    num_tracking_steps = gen_cfg.get('num_tracking_steps', 0)

    # If FiLM is active and no conditioner was provided, sample standard normal noise as condition
    if is_calib and cond_method == 'film' and conditioner is None:
        conditioner = torch.randn(num_samples, calib_cfg.get('num_params', 4), device=device)

    with torch.no_grad():
        training_noise_schedule, talpha_cum = get_noise_schedule(mod_cfg)
        T = mod_cfg['T']

        inference_noise_schedule = np.array(
            mod_cfg['inference_noise_schedule']) if fast_sampling else training_noise_schedule

        beta = inference_noise_schedule
        alpha = 1.0 - beta
        alpha_cum = np.cumprod(alpha)

        T_steps = []
        for s in range(len(inference_noise_schedule)):
            step_found = False
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                            talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                    T_steps.append(t + twiddle)
                    step_found = True
                    break

            if not step_found:
                if alpha_cum[s] >= talpha_cum[0]:
                    T_steps.append(0.0)
                else:
                    T_steps.append(float(T - 1))

        T_steps = np.array(T_steps, dtype=np.float32)

        signal = torch.randn(num_samples, full_len, device=device)
        noise_scale = torch.from_numpy(alpha_cum ** 0.5).float().to(device)

        if track_dir and num_tracking_steps > 0:
            last_10_steps = list(range(10))
            extra_5_steps = [199, 150, 100, 50, 20]
            track_indices = set(last_10_steps + extra_5_steps)
            current_track_path = track_dir if global_step == "inference" else os.path.join(track_dir,
                                                                                           f"step_{global_step}")
            os.makedirs(current_track_path, exist_ok=True)
        else:
            track_indices = set()

        for n in tqdm(range(len(alpha) - 1, -1, -1), desc="Denoising Progress"):
            # FIX: Determine current conditioner for step
            step_conditioner = signal[:, -4:] if (is_calib and cond_method == 'concat') else conditioner

            if n == len(alpha) - 1 and n in track_indices:
                signal_part = signal[0, :-4] if (is_calib and cond_method == 'concat') else signal[0]
                calib_part = signal[0, -4:] if (is_calib and cond_method == 'concat') else (
                    step_conditioner[0] if step_conditioner is not None else None)
                plot_diffusion_step(current_track_path, signal_part, noise_scale[n].item(), n, "T", max_val,
                                    calib_params=calib_part)

            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5

            t_tensor = torch.full((num_samples,), T_steps[n], device=device)
            predicted_noise = model(signal, t_tensor, conditioner=step_conditioner).squeeze(1)
            signal = c1 * (signal - c2 * predicted_noise)

            if n > 0:
                noise = torch.randn_like(signal)
                sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                signal += sigma * noise

            if n in track_indices and n != len(alpha) - 1:
                label = f"{n}"
                signal_part = signal[0, :-4] if (is_calib and cond_method == 'concat') else signal[0]
                calib_part = signal[0, -4:] if (is_calib and cond_method == 'concat') else (
                    step_conditioner[0] if step_conditioner is not None else None)
                plot_diffusion_step(current_track_path, signal_part, noise_scale[n].item(), n, label, max_val,
                                    calib_params=calib_part)

        signal = torch.clamp(signal, 0.0, 1.0)

    return signal, conditioner

def load_model_and_generate(config, device=torch.device('cuda'), track_dir=None, max_val=None):
    """
    Initializes the PyTorch model architecture, loads the saved `.pt` weights from disk,
    sets the model to evaluation mode, and hands it off to the generation engine.
    """
    weights_path = config['generation']['model_path']
    num_samples = config['generation']['num_samples_to_generate']

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = DiffWave(config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return run_reverse_diffusion_loop(model, config, num_samples, device, track_dir=track_dir, global_step="inference", max_val=max_val)


def execute_inference_pipeline(config, device, scaling_path, weights_path, is_calib, num_calib_params):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = os.path.splitext(os.path.basename(weights_path))[0]

    run_name = os.path.basename(os.path.dirname(weights_path))
    save_dir = f"results/{run_name}/{model_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    track_dir = os.path.join(save_dir, "denoising_process")
    os.makedirs(track_dir, exist_ok=True)

    max_val = np.load(scaling_path)

    # Generate full tensor of shape (num_samples, L + 4)
    signal, _ = load_model_and_generate(config=config, device=device, track_dir=track_dir, max_val=max_val)
    generated_scaled = signal.cpu().numpy()

    # --- Slice and Un-normalize ---
    if is_calib:
        gen_spectrum_norm = generated_scaled[:, :-num_calib_params]  # First L elements
        gen_coeffs_norm = generated_scaled[:, -num_calib_params:]    # Last 4 elements

        # Unnormalize spectrum and coefficients with their respective scaling logic
        generated_counts = unnormalize(gen_spectrum_norm, max_val)
        generated_counts = np.clip(np.round(generated_counts), a_min=0, a_max=None)
        gen_coeffs = unnormalize_coeffs(gen_coeffs_norm)
    else:
        generated_counts = unnormalize(generated_scaled, max_val)
        generated_counts = np.clip(np.round(generated_counts), a_min=0, a_max=None)
        gen_coeffs = None

    return generated_counts, gen_coeffs, save_dir, model_name


def main(args):
    config = load_config(args.exp, args.model)
    calib_cfg = config.get('calibration', {})
    is_calib = calib_cfg.get('enabled', False)
    num_calib_params = calib_cfg.get('num_params', 4)
    conditioning_method = calib_cfg.get('conditioning_method', 'concat')

    if is_calib:
        config['model']['signal_len'] += num_calib_params

    device = setup_device(config['hardware'], mode="inference")
    weights_path = config['generation']['model_path']
    num_samples = config['generation']['num_samples_to_generate']

    base_dir = config["paths"]["base_dir"]
    dataset_name = config["paths"]["dataset_name"]
    scaling_path = f"{base_dir}/scaling_factor_{dataset_name}.npy"

    print(f"Generating {num_samples} samples from {weights_path}...")

    generated_counts, gen_coeffs, save_dir, model_name = execute_inference_pipeline(
        config, device, scaling_path, weights_path, is_calib, num_calib_params
    )

    if gen_coeffs is not None:
        plot_and_save_spectra(generated_counts, save_dir, model_name, gen_coeffs)
        coeff_path = os.path.join(save_dir, f"all_{num_samples}_coeffs.npy")
        np.save(coeff_path, gen_coeffs)
    else:
        plot_and_save_spectra(generated_counts, save_dir, model_name)

    data_path = os.path.join(save_dir, f"all_{num_samples}_samples.npy")
    np.save(data_path, generated_counts)
    print(f"Successfully saved all {num_samples} images and data to: {save_dir}/")


if __name__ == '__main__':
    parser = ArgumentParser(description='Runs inference for 1-D DiffWave')
    parser.add_argument("--exp", type=str, default="configs/experiments/diffwave_base.yaml", help="Path to experiment config")
    parser.add_argument("--model", type=str, default="configs/models/diffwave.yaml", help="Path to model config")
    main(parser.parse_args())