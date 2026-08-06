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

from utils.helpers import load_config, setup_device, plot_and_save_spectra, unnormalize
from models.diffwave import DiffWave


def generate_samples(model, config, num_samples, device, track_dir=None, global_step=None, max_val=None):
    """Core sampling logic for DiffWave with linear step tracking."""
    gen_cfg = config['generation']
    mod_cfg = config['model']

    fast_sampling = gen_cfg.get('fast_sampling', True)
    signal_len = mod_cfg['signal_len']
    num_tracking_steps = gen_cfg.get('num_tracking_steps', 0)

    with torch.no_grad():
        T = mod_cfg['T']
        beta_init = mod_cfg['beta_init']
        beta_final = mod_cfg['beta_final']

        # Define noise variance schedules (beta)
        training_noise_schedule = np.linspace(beta_init, beta_final, T, dtype=np.float32)
        inference_noise_schedule = np.array(
            mod_cfg['inference_noise_schedule']) if fast_sampling else training_noise_schedule

        # alpha = 1 - beta. alpha_cum is alpha_bar (cumulative product)
        # These track the exact ratio of clean signal to noise at any given step
        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        # Map inference steps to fractional training steps (interpolation for fast sampling)
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

            # --- NEW: Fallback for floating-point rounding misses at the edges ---
            if not step_found:
                if alpha_cum[s] >= talpha_cum[0]:
                    T_steps.append(0.0)
                else:
                    T_steps.append(float(T - 1))

        T_steps = np.array(T_steps, dtype=np.float32)
        # Start inference with pure Gaussian noise (x_T)
        signal = torch.randn(num_samples, signal_len, device=device)
        noise_scale = torch.from_numpy(alpha_cum ** 0.5).float().to(device)

        if track_dir and num_tracking_steps > 0:
            # 1. The last 10 steps of generation (indices 9 down to 0)
            last_10_steps = list(range(10))

            # 2. 5 extra steps further out (you can change these numbers to whatever you want)
            # For example: step 199 (start), 150, 100, 50, and 20
            extra_5_steps = [199, 150, 100, 50, 20]
            track_indices = set(last_10_steps + extra_5_steps)

            if global_step == "inference":
                current_track_path = track_dir
            else:
                current_track_path = os.path.join(track_dir, f"step_{global_step}")

            os.makedirs(current_track_path, exist_ok=True)
        else:
            track_indices = set()

        # Reverse diffusion loop: gradually denoise from step T down to 0
        # <-- Wrapped range() with tqdm here -->
        for n in tqdm(range(len(alpha) - 1, -1, -1), desc="Denoising Progress"):
            if n == len(alpha) - 1 and n in track_indices:
                _save_snapshot(current_track_path, signal[0], noise_scale[n].item(), n, "initial_T", max_val)

            # c1 and c2 are constants for the reverse step mean equation
            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5

            # Predict the noise using the model, then subtract it to estimate the cleaner signal
            signal = c1 * (signal - c2 * model(signal, torch.tensor([T_steps[n]], device=signal.device)).squeeze(1))

            if n > 0:
                # Inject a calculated amount of new variance (Langevin noise) back in
                # This prevents the signal from collapsing to a deterministic average
                noise = torch.randn_like(signal)
                sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                signal += sigma * noise

            if n in track_indices and n != len(alpha) - 1:
                label = "final_0" if n == 0 else f"mid_{n}"
                _save_snapshot(current_track_path, signal[0], noise_scale[n].item(), n, label, max_val)

        # Force final output x_0 to match the dataset's specific normalized range
        signal = torch.clamp(signal, 0.0, 1.0)

    return signal


def _save_snapshot(track_path, single_signal, noise_scale_val, loop_idx, label, max_val=None):
    """Helper to generate visual plots of specific timesteps."""
    sig_np = single_signal.cpu().numpy()

    # --- NEW: Unnormalize and clean the signal if max_val is provided ---
    if max_val is not None:
        sig_np = unnormalize(sig_np, max_val)
        sig_np = np.clip(np.round(sig_np), a_min=0, a_max=None)
        y_label = "Counts (Log Scale)"
    else:
        y_label = "Amplitude (Normalized)"
    # --------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sig_np, color='blue', linewidth=1)

    # Set to log scale if we unnormalized, to match your standard plots
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


def predict(config, device=torch.device('cuda'), track_dir=None, max_val=None):
    weights_path = config['generation']['model_path']
    num_samples = config['generation']['num_samples_to_generate']

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = DiffWave(config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # --- NEW: Pass max_val into generate_samples ---
    return generate_samples(model, config, num_samples, device, track_dir=track_dir, global_step="inference", max_val=max_val)


def generate_and_process_samples(config, device, scaling_path, weights_path):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = os.path.splitext(os.path.basename(weights_path))[0]
    run_name = os.path.basename(os.path.dirname(weights_path))
    save_dir = f"results/{run_name}/{model_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    track_dir = os.path.join(save_dir, "denoising_process")
    os.makedirs(track_dir, exist_ok=True)

    # --- NEW: Load max_val early and pass it into predict ---
    max_val = np.load(scaling_path)
    signal = predict(config=config, device=device, track_dir=track_dir, max_val=max_val)
    # --------------------------------------------------------

    generated_scaled = signal.cpu().numpy()
    generated_counts = unnormalize(generated_scaled, max_val)
    generated_counts = np.clip(np.round(generated_counts), a_min=0, a_max=None)

    return generated_counts, save_dir, model_name


def main(args):
    # Load and merge configuration blocks
    config = load_config(args.exp, args.model)

    device = setup_device(config['hardware'])
    weights_path = config['generation']['model_path']
    num_samples = config['generation']['num_samples_to_generate']
    scaling_path = config['paths']['scaling_factor_path']

    print(f"Generating {num_samples} samples from {weights_path}...")

    generated_counts, save_dir, model_name = generate_and_process_samples(
        config, device, scaling_path, weights_path
    )

    plot_and_save_spectra(generated_counts, save_dir, model_name)

    data_path = os.path.join(save_dir, f"all_{num_samples}_samples.npy")
    np.save(data_path, generated_counts)
    print(f"Successfully saved all {num_samples} images and data to: {save_dir}/")


if __name__ == '__main__':
    parser = ArgumentParser(description='Runs unconditional inference for 1-D DiffWave')
    parser.add_argument("--exp", type=str, default="configs/experiments/diffwave_base.yaml",
                        help="Path to the experiment config")
    parser.add_argument("--model", type=str, default="configs/models/diffwave.yaml", help="Path to the model config")
    main(parser.parse_args())