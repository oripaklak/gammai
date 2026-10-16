import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from utils.helpers import normalize, normalize_coeffs


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, config):
        super().__init__()
        calib_cfg = config.get('calibration', {})
        self.is_calib = calib_cfg.get('enabled', False)
        self.cond_method = calib_cfg.get('conditioning_method', 'concat')

        # Normalize input format (handles str, Path, or legacy 1-element list)
        if isinstance(data_path, (list, tuple)):
            path_obj = Path(data_path[0])
        else:
            path_obj = Path(data_path)

        # Determine dataset directory and spectrum matrix path
        if path_obj.is_file():
            self.dataset_dir = path_obj.parent
            spectrum_path = path_obj
        else:
            self.dataset_dir = path_obj
            spectrum_path = self.dataset_dir / "spectrum_matrix.npy"

        if not spectrum_path.exists():
            raise FileNotFoundError(f"Missing spectrum matrix at: {spectrum_path}")

        # Load raw spectrum matrix (N x L)
        raw_matrix = np.load(spectrum_path).astype(np.float32)
        log_matrix = np.log1p(raw_matrix)
        self.max_val = np.max(log_matrix)
        self.matrix = normalize(raw_matrix, self.max_val)

        if self.is_calib:
            coeff_path = self.dataset_dir / "coeff_matrix.npy"
            bounds_path = self.dataset_dir / "global_calib_bounds.json"

            if not coeff_path.exists():
                raise FileNotFoundError(f"Missing coeff matrix at: {coeff_path}")

            raw_coeffs = np.load(coeff_path).astype(np.float32)

            # Normalize using dataset-specific calibration bounds
            try:
                self.coeffs = normalize_coeffs(raw_coeffs, bounds_path=str(bounds_path))
            except TypeError:
                self.coeffs = normalize_coeffs(raw_coeffs)

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        signal = self.matrix[idx]  # Shape: (L,)

        if self.is_calib:
            coeff = self.coeffs[idx]  # Shape: (4,)
            if self.cond_method == 'concat':
                signal = np.concatenate([signal, coeff], axis=0)  # Shape: (L+4,)
                return {'signal': signal}
            else:
                return {
                    'signal': signal,  # Shape: (L,)
                    'conditioner': coeff  # Shape: (4,)
                }

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


def from_path(data_dir, config, is_distributed=False):
    """
    Creates a DataLoader for SignalDataset.

    Args:
        data_dir: Dataset folder path (e.g., 'data/Baq'), file path, or None.
        config: Configuration dictionary.
        is_distributed: Whether PyTorch DDP sampler should be attached.
    """
    paths_cfg = config.get('paths', {})
    base_dir = paths_cfg.get('base_dir', 'data')
    dataset_name = paths_cfg.get('dataset_name', '')

    # Auto-construct dataset directory if data_dir is None or points to root 'data'
    if data_dir is None or data_dir == base_dir:
        data_dir = os.path.join(base_dir, dataset_name)

    dataset = SignalDataset(data_dir, config)
    global_seed = config.get('training', {}).get('seed', 42)

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