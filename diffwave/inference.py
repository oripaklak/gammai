import numpy as np
import os
import torch

from argparse import ArgumentParser

from diffwave.params import AttrDict, params as base_params
from diffwave.model import DiffWave

models = {}


def predict(model_dir=None, params=None, device=torch.device('cuda'), fast_sampling=False):
    # Lazy load model.
    if not model_dir in models:
        if os.path.exists(f'{model_dir}/weights.pt'):
            checkpoint = torch.load(f'{model_dir}/weights.pt')
        else:
            checkpoint = torch.load(model_dir)
        model = DiffWave(AttrDict(base_params)).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        models[model_dir] = model

    model = models[model_dir]
    model.params.override(params)
    with torch.no_grad():
        # Change in notation from the DiffWave paper for fast sampling.
        # DiffWave paper -> Implementation below
        # --------------------------------------
        # alpha -> talpha
        # beta -> training_noise_schedule
        # gamma -> alpha
        # eta -> beta
        training_noise_schedule = np.array(model.params.noise_schedule)
        inference_noise_schedule = np.array(
            model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

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

        # Unconditional 1D signal generation (Replacing audio logic)
        signal = torch.randn(1, params.signal_len, device=device)
        noise_scale = torch.from_numpy(alpha_cum ** 0.5).float().unsqueeze(1).to(device)

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5

            # Pass None in place of the spectrogram parameter
            signal = c1 * (signal - c2 * model(signal, torch.tensor([T[n]], device=signal.device), None).squeeze(1))

            if n > 0:
                noise = torch.randn_like(signal)
                sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                signal += sigma * noise

            signal = torch.clamp(signal, -1.0, 1.0)

    return signal


def main(args):
    # Run the prediction (spectrogram logic completely removed)
    signal = predict(model_dir=args.model_dir, fast_sampling=args.fast, params=base_params)

    # Save the resulting tensor as a NumPy array
    np.save(args.output, signal.cpu().squeeze().numpy())
    print(f"Signal successfully generated and saved to {args.output}")


if __name__ == '__main__':
    parser = ArgumentParser(description='runs unconditional inference for 1-D DiffWave')
    parser.add_argument('model_dir',
                        help='directory containing a trained model (or full path to weights.pt file)')
    parser.add_argument('--output', '-o', default='output.npy',
                        help='output file name (e.g., output.npy)')
    parser.add_argument('--fast', '-f', action='store_true',
                        help='fast sampling procedure')
    main(parser.parse_args())