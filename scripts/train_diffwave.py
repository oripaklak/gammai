import argparse
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

from torch.cuda import device_count
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

# Add the project root to the Python path to import from models/ and utils/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import from_path
from utils.helpers import load_config, set_seed
from utils.metrics import calculate_frechet_distance, calculate_polynomial_mmd
from models.diffwave import DiffWave
from inference_diffwave import generate_samples

# ==============================================================================
# Core Training Logic
# ==============================================================================
def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


class DiffWaveLearner:
    def __init__(self, model_dir, model, dataset, optimizer, config):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.config = config

        self.train_cfg = config['training']
        self.model_cfg = config['model']
        self.hw_cfg = config['hardware']
        self.track_cfg = config['tracking']['wandb']

        self.autocast = torch.amp.autocast('cuda', enabled=self.hw_cfg.get('fp16', False))
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.hw_cfg.get('fp16', False))

        self.step = 0
        self.is_master = True

        beta = np.array(self.model_cfg['noise_schedule'])
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))
        self.loss_fn = nn.L1Loss()

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in
                          self.optimizer.state_dict().items()},
            'config': self.config,
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self):
        device = next(self.model.parameters()).device

        for epoch in range(self.train_cfg['epochs']):
            iterator = tqdm(self.dataset,
                            desc=f"Epoch {epoch + 1}/{self.train_cfg['epochs']}") if self.is_master else self.dataset

            for features in iterator:
                features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                loss = self.train_step(features)

                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')

                if self.is_master:
                    if self.step % self.train_cfg['print_freq'] == 0:
                        self._write_summary(self.step, features, loss)

                    # --- DiffWave Fast Sampling Evaluation ---
                    if self.step > 0 and self.step % self.train_cfg['checkpoint_freq'] == 0:
                        self.save_to_checkpoint()

                        # Use the imported generation logic!
                        fake_signal = generate_samples(
                            model=self.model,
                            config=self.config,
                            num_samples=16,
                            device=device
                        )
                        fake_np = fake_signal.cpu().numpy()

                        # Grab up to 16 real samples from the current batch for comparison
                        real_np = features['signal'][:16].cpu().numpy()

                        try:
                            fd_score = calculate_frechet_distance(real_np, fake_np)
                            kid_score = calculate_polynomial_mmd(real_np, fake_np)

                            if self.track_cfg['enabled']:
                                wandb.log({
                                    "eval/FID": fd_score,
                                    "eval/KID": kid_score
                                }, step=self.step)

                            print(f"\n--> Eval Step {self.step} | FD: {fd_score:.4f} | KD: {kid_score:.4f}")
                        except Exception as e:
                            print(f"\nWarning: Failed to calculate evaluation metrics. Error: {e}")

                self.step += 1

    def train_step(self, features):
        for param in self.model.parameters():
            param.grad = None

        signal = features['signal']

        N, T = signal.shape
        device = signal.device
        self.noise_level = self.noise_level.to(device)

        with self.autocast:
            t = torch.randint(0, len(self.model_cfg['noise_schedule']), [N], device=signal.device)
            noise_scale = self.noise_level[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale ** 0.5
            noise = torch.randn_like(signal)
            noisy_signal = noise_scale_sqrt * signal + (1.0 - noise_scale) ** 0.5 * noise

            predicted = self.model(noisy_signal, t, None)
            loss = self.loss_fn(noise, predicted.squeeze(1))

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.get('max_grad_norm') or 1e9)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def _write_summary(self, step, features, loss):
        fig, ax = plt.subplots()
        ax.plot(features['signal'][0].cpu().numpy())
        ax.set_title(f'DiffWave Training Signal at step {step}')

        current_epoch = step // len(self.dataset)

        if self.track_cfg['enabled']:
            wandb.log({
                'train/epoch': current_epoch,
                'train/global_step': step,
                'train/loss_diffwave': loss,
                'train/grad_norm_diffwave': self.grad_norm,
                'feature/signal_diffwave': wandb.Image(fig)
            }, step=step)

        plt.close(fig)


# ==============================================================================
# Execution & Distribution Handlers
# ==============================================================================
def _train_impl(replica_id, model, dataset, config, run_dir):
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    learner = DiffWaveLearner(run_dir, model, dataset, opt, config)
    learner.is_master = (replica_id == 0)

    if learner.is_master and config['tracking']['wandb']['enabled']:
        wandb.init(
            project=config['experiment']['project_name'],
            name=os.path.basename(run_dir),
            config=config
        )

    learner.restore_from_checkpoint()
    learner.train()

    if learner.is_master and config['tracking']['wandb']['enabled']:
        wandb.finish()


def train(config, run_dir):
    dataset = from_path([config['paths']['data_path']], config)
    model = DiffWave(config).cuda()
    _train_impl(0, model, dataset, config, run_dir)


def train_distributed(replica_id, replica_count, port, config, run_dir):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)

    dataset = from_path([config['paths']['data_path']], config, is_distributed=True)
    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    model = DiffWave(config).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id])
    _train_impl(replica_id, model, dataset, config, run_dir)


def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]


def main():
    parser = argparse.ArgumentParser(description='Train 1-D DiffWave Model')
    parser.add_argument("--exp", type=str, default="configs/experiments/diffwave_base.yaml", help="Path to the experiment config")
    parser.add_argument("--model", type=str, default="configs/models/diffwave.yaml", help="Path to the model config")
    args = parser.parse_args()

    # Load configuration blocks
    config = load_config(args.exp, args.model)
    train_cfg = config["training"]
    hw_cfg = config["hardware"]
    path_cfg = config["paths"]

    # Set Random Seed for Reproducibility
    set_seed(train_cfg["seed"])

    # Auto-generate a unique checkpoint directory
    run_name = f"diff_bs{train_cfg['batch_size']}_lr{train_cfg['learning_rate']}_ep{train_cfg['epochs']}"
    run_dir = os.path.join(path_cfg['checkpoints_dir'], run_name)

    # Determine replica count based on YAML override or available GPUs
    replica_count = hw_cfg['num_gpus'] if hw_cfg['num_gpus'] > 0 else device_count()

    if replica_count > 1:
        if train_cfg['batch_size'] % replica_count != 0:
            raise ValueError(f"Batch size {train_cfg['batch_size']} is not evenly divisble by # GPUs {replica_count}.")
        config['training']['batch_size'] = train_cfg['batch_size'] // replica_count
        port = _get_free_port()
        spawn(train_distributed, args=(replica_count, port, config, run_dir), nprocs=replica_count, join=True)
    else:
        train(config, run_dir)


if __name__ == '__main__':
    main()