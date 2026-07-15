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
        self.filenames = []
        for path in paths:
            # Assuming your 1-D signals are saved as standard NumPy arrays
            self.filenames += glob(f'{path}/**/*.npy', recursive=True)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        signal_filename = self.filenames[idx]
        # Load the 1-D vector
        signal = np.load(signal_filename)
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