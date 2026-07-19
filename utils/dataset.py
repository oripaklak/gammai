import numpy as np
import os
import torch
from torch.utils.data.distributed import DistributedSampler


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, paths, config):
        super().__init__()

        # Load the raw 2D matrix
        raw_matrix = np.load(paths[0]).astype(np.float32)

        print(f"\n[INFO] Loaded dataset from {paths[0]} with shape: {raw_matrix.shape}")
        if len(raw_matrix.shape) != 2:
            raise ValueError(f"ERROR: Expected a 2D matrix (N x L), but got shape {raw_matrix.shape}.")

        # Apply normalization logic
        log_matrix = np.log1p(raw_matrix)
        self.max_val = np.max(log_matrix)
        self.matrix = log_matrix / self.max_val

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        signal = self.matrix[idx]
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
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=Collator(config).collate,
        shuffle=not is_distributed,
        num_workers=0,  # Changed to 0 since the matrix is fully loaded in memory
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True
    )