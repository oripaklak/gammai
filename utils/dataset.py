import numpy as np
import os
import torch
from torch.utils.data.distributed import DistributedSampler
from utils.helpers import normalize, normalize_coeffs


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, paths, config):
        super().__init__()
        self.is_calib = config.get('experiment', {}).get('calibration', False)

        # 1. Load the raw 2D spectrum matrix
        raw_matrix = np.load(paths[0]).astype(np.float32)

        print(f"\n[INFO] Loaded dataset from {paths[0]} with shape: {raw_matrix.shape}")
        if len(raw_matrix.shape) != 2:
            raise ValueError(f"ERROR: Expected a 2D matrix (N x L), but got shape {raw_matrix.shape}.")

        # Find the global max_val required for normalization
        log_matrix = np.log1p(raw_matrix)
        self.max_val = np.max(log_matrix)

        # Apply normalization using the helper function
        self.matrix = normalize(raw_matrix, self.max_val)

        # 2. Conditionally load and normalize the coefficients
        if self.is_calib:
            coeff_path = paths[0].replace("spectrum_matrix", "coeff_matrix")

            if not os.path.exists(coeff_path):
                raise FileNotFoundError(f"Calibration enabled but coeff matrix missing at: {coeff_path}")

            raw_coeffs = np.load(coeff_path).astype(np.float32)
            print(f"[INFO] Loaded coefficients from {coeff_path} with shape: {raw_coeffs.shape}")

            # Apply Min-Max scaling to the entire N x 4 matrix at once
            self.coeffs = normalize_coeffs(raw_coeffs)

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        signal = self.matrix[idx]

        # 3. Concatenate L and 4 into L+4 dynamically
        if self.is_calib:
            coeff = self.coeffs[idx]
            signal = np.concatenate([signal, coeff], axis=0)

        return {
            'signal': signal
        }


class Collator:
    def __init__(self, config):
        self.config = config

    def collate(self, minibatch):
        signal_batch = np.stack([record['signal'] for record in minibatch])
        return {
            'signal': torch.from_numpy(signal_batch).float(),
        }


def from_path(data_dirs, config, is_distributed=False):
    dataset = SignalDataset(data_dirs, config)

    # Extract the global seed from your config (assuming it is stored there)
    global_seed = config['training'].get('seed', 42)

    # Pass the seed to the sampler to ensure it syncs with your global randomness
    sampler = DistributedSampler(dataset, seed=global_seed) if is_distributed else None

    # Optional: explicitly seed the DataLoader's internal generator for the non-distributed shuffle
    g = torch.Generator()
    g.manual_seed(global_seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=Collator(config).collate,
        shuffle=(not is_distributed),
        num_workers=0,  # Matrix is fully loaded in memory
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        generator=g if not is_distributed else None  # Locks the standard shuffle
    )