import numpy as np
import os
import random
import torch
import torch.nn.functional as F

from glob import glob
from torch.utils.data.distributed import DistributedSampler


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super().__init__()

        # 1. Load the raw 2D matrix
        raw_matrix = np.load(paths[0]).astype(np.float32)

        print(f"\n[INFO] Loaded dataset from {paths[0]} with shape: {raw_matrix.shape}")
        if len(raw_matrix.shape) != 2:
            raise ValueError(f"ERROR: Expected a 2D matrix (N x L), but got shape {raw_matrix.shape}.")

        # 2. Apply your exact normalization logic
        log_matrix = np.log1p(raw_matrix)
        max_val = np.max(log_matrix)
        self.matrix = log_matrix / max_val

    def __len__(self):
        # Returns the number of spectrums (N)
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        # Grabs a single spectrum vector based on the index
        signal = self.matrix[idx]
        return {
            'signal': signal
        }


class Collator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):
        # Since all signals are guaranteed to be exactly L = 2**14 long,
        # we bypass all length checks and stack them directly into a batch.
        signal_batch = np.stack([record['signal'] for record in minibatch])

        return {
            # Convert to a float32 tensor which DiffWave expects
            'signal': torch.from_numpy(signal_batch).float(),
        }

def from_path(data_dirs, params, is_distributed=False):
    dataset = SignalDataset(data_dirs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).collate,
        shuffle=not is_distributed,
        num_workers=os.cpu_count(),
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True)