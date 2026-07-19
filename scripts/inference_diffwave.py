import numpy as np
import os
import sys
import datetime
import torch
from argparse import ArgumentParser

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helpers import load_config, setup_device, plot_and_save_spectra
from models.diffwave import DiffWave


def generate_samples(model, config, num_samples, device):
    """Core sampling logic for DiffWave"""
    model.eval()
    gen_cfg = config['generation']
    mod_cfg = config['model']

    fast_sampling = gen_cfg['fast_sampling']
    signal_len = mod_cfg['signal_len']

    with torch.no_grad():
        training_noise_schedule = np.array(mod_cfg['noise_schedule'])
        inference_noise_schedule = np.array(
            mod_cfg['inference_noise_schedule']) if fast_sampling else training_noise_schedule

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                            talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                    T.append(t + twiddle)
                    break
        T = np.array(T, dtype=np.float32)

        signal = torch.randn(num_samples, signal_len, device=device)

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5

            signal = c1 * (signal - c2 * model(signal, torch.tensor([T[n]], device=signal.device), None).squeeze(1))

            if n > 0:
                noise = torch.randn_like(signal)
                sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                signal += sigma * noise

        signal = torch.clamp(signal, -1.0, 1.0)

    model.train()  # Reset to train mode
    return signal


def predict(config, device=torch.device('cuda')):
    """Loads weights from disk and calls the reusable generation loop."""
    weights_path = config['generation']['weights_to_use']
    num_samples = config['generation']['num_samples_to_generate']

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    model = DiffWave(config).to(device)
    model.load_state_dict(checkpoint['model'])

    # Call the isolated generation logic
    return generate_samples(model, config, num_samples, device)


def main(args):
    # Load and merge configuration blocks
    config = load_config(args.exp, args.model)

    device = setup_device(config['hardware'])
    weights_path = config['generation']['weights_to_use']
    num_samples = config['generation']['num_samples_to_generate']
    scaling_path = config['paths']['scaling_factor_path']

    print(f"Generating {num_samples} samples from {weights_path}...")

    # 1. Generate normalized signals
    signal = predict(config=config, device=device)
    generated_scaled = signal.cpu().numpy()

    # 2. Reverse the Normalization
    max_val = np.load(scaling_path)
    generated_log = generated_scaled * max_val
    generated_counts = np.expm1(generated_log)
    generated_counts = np.clip(np.round(generated_counts), a_min=0, a_max=None)

    # 3. Setup Directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = os.path.splitext(os.path.basename(weights_path))[0]
    run_name = os.path.basename(os.path.dirname(weights_path))
    save_dir = f"results/{run_name}/{model_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 4. Plot and Save
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