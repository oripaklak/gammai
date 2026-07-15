import numpy as np
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

from diffwave.params import params


def transform(filename):
    # Load your 1-D signal
    signal = np.load(filename)

    # 1. Validate length matches params.signal_len (16000)
    if len(signal) != params.signal_len:
        raise ValueError(f'Invalid length in {filename}: expected {params.signal_len}, got {len(signal)}.')

    # 2. Convert to float32 (required for PyTorch/DiffWave)
    signal = signal.astype(np.float32)

    # 3. Optional: Clamp or Normalize to [-1.0, 1.0]
    # Diffusion models typically expect inputs in this range.
    # Uncomment or modify the line below based on your gamma spectrogram data distribution:
    # signal = np.clip(signal, -1.0, 1.0)

    # Save the standardized file (overwriting the original, or you can change the path)
    np.save(filename, signal)


def main(args):
    # Search for .npy files instead of .wav files
    filenames = glob(f'{args.dir}/**/*.npy', recursive=True)

    if not filenames:
        print(f"No .npy files found in {args.dir}")
        return

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(transform, filenames), desc='Validating & Preprocessing', total=len(filenames)))


if __name__ == '__main__':
    parser = ArgumentParser(description='validates and prepares 1-D .npy files for DiffWave')
    parser.add_argument('dir',
                        help='directory containing .npy signal files for training')
    main(parser.parse_args())