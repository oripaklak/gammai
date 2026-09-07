import numpy as np
import os
import torch
from torch.utils.data.distributed import DistributedSampler
from utils.helpers import normalize, normalize_coeffs


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, paths, config):
        super().__init__()
        calib_cfg = config.get('calibration', {})
        self.is_calib = calib_cfg.get('enabled', False)
        self.cond_method = calib_cfg.get('conditioning_method', 'concat')

        # Load raw spectrum matrix (N x L)
        raw_matrix = np.load(paths[0]).astype(np.float32)
        log_matrix = np.log1p(raw_matrix)
        self.max_val = np.max(log_matrix)
        self.matrix = normalize(raw_matrix, self.max_val)

        if self.is_calib:
            coeff_path = paths[0].replace("spectrum_matrix", "coeff_matrix")
            if not os.path.exists(coeff_path):
                raise FileNotFoundError(f"Missing coeff matrix at: {coeff_path}")
            raw_coeffs = np.load(coeff_path).astype(np.float32)
            self.coeffs = normalize_coeffs(raw_coeffs)

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        signal = self.matrix[idx]  # Shape: (L,)

        if self.is_calib:
            coeff = self.coeffs[idx]  # Shape: (4,)
            # ALWAYS concatenate to create an (L + 4,) sequence
            signal = np.concatenate([signal, coeff], axis=0)

            item = {'signal': signal}
            if self.cond_method == 'film':
                item['conditioner'] = coeff  # Separate tensor for FiLM layer modulation
            return item

        return {'signal': signal}


class Collator:
    def __init__(self, config):
        self.config = config

    def collate(self, minibatch):
        signal_batch = np.stack([record['signal'] for record in minibatch])
        batch = {
            'signal': torch.from_numpy(signal_batch).float()
        }
        if 'conditioner' in minibatch[0]:
            cond_batch = np.stack([record['conditioner'] for record in minibatch])
            batch['conditioner'] = torch.from_numpy(cond_batch).float()

        return batch


def from_path(data_dirs, config, is_distributed=False):
    dataset = SignalDataset(data_dirs, config)
    global_seed = config['training'].get('seed', 42)

    sampler = DistributedSampler(dataset, seed=global_seed) if is_distributed else None

    g = torch.Generator()
    g.manual_seed(global_seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=Collator(config).collate,
        shuffle=(not is_distributed),
        num_workers=0,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        generator=g if not is_distributed else None
    )