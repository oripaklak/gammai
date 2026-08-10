import argparse
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import torch.distributed as dist

from torch.cuda import device_count
from torch.multiprocessing import spawn
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

# Add the project root to the Python path to import from models/ and utils/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import from_path
from utils.helpers import load_config, set_seed, count_parameters, format_param_count, register_graceful_shutdown, get_noise_schedule
from utils.metrics import calculate_frechet_distance, calculate_polynomial_mmd
from models.diffwave import DiffWave
from inference_diffwave import run_reverse_diffusion_loop

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
        self.saved_checkpoints = []

        # --- Tracking Directories Setup ---
        self.track_dir = os.path.join(model_dir, "tracked_signals")
        self.track_dir_eval = os.path.join(self.track_dir, "eval")
        self.track_dir_train = os.path.join(self.track_dir, "train")

        if self.is_master:
            os.makedirs(self.track_dir_eval, exist_ok=True)
            os.makedirs(self.track_dir_train, exist_ok=True)

        # Unpack both beta and noise_level (alpha_bar) directly
        beta, noise_level = get_noise_schedule(self.model_cfg)

        self.noise_level = torch.tensor(noise_level.astype(np.float32))
        self.loss_fn = nn.L1Loss()

        # Load scaling factor for un-normalizing x_0 tracks
        self.max_val = self.dataset.dataset.max_val

        # Save max_val to the scaling factor path specified in config
        if self.is_master:
            scaling_path = self.config["paths"]["scaling_factor_path"]
            os.makedirs(os.path.dirname(scaling_path), exist_ok=True)
            np.save(scaling_path, self.max_val)
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
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

        # Restore RNG states
        if 'torch_rng_state' in state_dict:
            torch.set_rng_state(state_dict['torch_rng_state'])
            torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
            np.random.set_state(state_dict['numpy_rng_state'])
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

            # --- Rolling Checkpoint Logic ---
            self.saved_checkpoints.append(save_name)
            keep_n = self.train_cfg.get("keep_last_n_checkpoints", 5)

            if len(self.saved_checkpoints) > keep_n:
                oldest_ckpt = self.saved_checkpoints.pop(0)
                if os.path.exists(oldest_ckpt):
                    os.remove(oldest_ckpt)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt', weights_only=False)
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def load_pretrained_weights(self, filepath):
        """Loads only the model weights for fine-tuning, ignoring optimizer and step state."""
        try:
            # Load checkpoint to CPU first to avoid VRAM spikes
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

            # Extract just the model state (handles both dicts and raw states)
            if 'model' in checkpoint:
                model_state = checkpoint['model']
            else:
                model_state = checkpoint

            # Load into the model (unwrapping DDP if necessary)
            if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
                self.model.module.load_state_dict(model_state)
            else:
                self.model.load_state_dict(model_state)

            return True
        except Exception as e:
            if self.is_master:
                print(f"Failed to load pre-trained weights from {filepath}. Error: {e}")
            raise

    def evaluate(self):
        """Generates fast samples and calculates FID/KID metrics using optimized batch sizes."""
        device = next(self.model.parameters()).device
        num_samples_to_generate = 256

        # --- 0. Isolate RNG State (Crucial for perfect resuming!) ---
        torch_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()

        # NOTE: 128 might cause CUDA Out of Memory. If it crashes, lower this to 64 or 32.
        eval_batch_size = 64

        print(f"\nGenerating {num_samples_to_generate} samples for evaluation (Eval Batch Size: {eval_batch_size})...")

        model_to_eval = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_eval.eval()

        # --- 1. Collect EXACTLY 256 Real Samples ---
        all_real_data = []
        real_collected = 0

        for eval_batch in self.dataset:
            real_batch = eval_batch['signal']
            all_real_data.append(real_batch.cpu().numpy())
            real_collected += real_batch.size(0)

            if real_collected >= num_samples_to_generate:
                break

        # Slice to exactly 256
        real_np = np.concatenate(all_real_data, axis=0)[:num_samples_to_generate]

        # --- 2. Generate EXACTLY 256 Fake Samples ---
        all_fake_data = []
        generated_so_far = 0

        with torch.no_grad():
            while generated_so_far < num_samples_to_generate:
                # This ensures the last batch doesn't generate more than we need
                current_batch_size = min(eval_batch_size, num_samples_to_generate - generated_so_far)

                # Track signals only on the first batch
                current_track_dir = self.track_dir_eval if generated_so_far == 0 else None

                fake_signal = run_reverse_diffusion_loop(
                    model=model_to_eval,
                    config=self.config,
                    num_samples=current_batch_size,
                    device=device,
                    track_dir=current_track_dir,
                    global_step=self.step,
                    max_val=self.max_val
                )

                all_fake_data.append(fake_signal.cpu().numpy())
                generated_so_far += current_batch_size

        fake_np = np.concatenate(all_fake_data, axis=0)

        # --- 3. Calculate Metrics ---
        try:
            fd_score = calculate_frechet_distance(real_np, fake_np)
            kid_score = calculate_polynomial_mmd(real_np, fake_np)

            if self.track_cfg['enabled']:
                import wandb
                wandb.log({
                    "eval/FID": fd_score,
                    "eval/KID": kid_score
                }, step=self.step)

            print(f"--> Eval Step {self.step} | FD: {fd_score:.4f} | KD: {kid_score:.4f}")
        except Exception as e:
            print(f"\nWarning: Failed to calculate evaluation metrics. Error: {e}")

        # --- 4. Restore RNG State ---
        torch.set_rng_state(torch_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

    def track_forward_process(self, clean_signal, noisy_signal, noise, t_val, global_step):
        """Plots the actual noisy signal vs original clean signal for comparison."""
        T_total = self.model_cfg['T']

        # Python indices: 0 -> x_1, 99 -> x_100, 199 -> x_200 (for T=200)
        interesting_t_values = [0, T_total // 2 - 1, T_total - 1]

        if t_val not in interesting_t_values:
            return

        # Map Python's 0-index back to the paper's 1-indexed math
        math_step = t_val + 1

        # --- 1. Track x_0 (The Pure Clean Signal) ---
        # We save this standalone plot whenever x_1 is triggered
        if math_step == 1:
            x0_path = os.path.join(self.track_dir_train, "x_0")
            os.makedirs(x0_path, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(clean_signal[0].cpu().numpy(), color="blue")
            ax.set_title(f"Original Clean Signal (x_0)\nCaptured at Global Step: {global_step}")
            ax.set_ylabel("Amplitude (Normalized)")
            plt.tight_layout()
            plt.savefig(os.path.join(x0_path, f"clean_signal_step{global_step}.png"))
            plt.close(fig)

        # --- 2. Track the Noisy Steps (x_1, x_100, x_200) ---
        current_track_path = os.path.join(self.track_dir_train, f"x_{math_step}")
        os.makedirs(current_track_path, exist_ok=True)

        # Extract numpy arrays
        clean_np = clean_signal[0].cpu().numpy()
        noisy_np = noisy_signal[0].cpu().numpy()

        # Plot Clean vs Noisy using the paper's exact notation
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(clean_np, label="x_0 (Original Clean Signal)", alpha=0.75, color="blue")
        ax.plot(noisy_np, label=f"x_{math_step} (Noisy Signal)", alpha=0.75, color="red", linestyle="--")

        ax.set_title(f"Forward Pass Signal Comparison | Step: x_{math_step}\nCaptured at Global Step: {global_step}")
        ax.set_ylabel("Amplitude (Normalized)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(current_track_path, f"noisy_vs_clean_step{global_step}.png"))
        plt.close(fig)

    def train(self):
        device = next(self.model.parameters()).device

        # --- 1. Find WandB's Last Logged Step ---
        wandb_catchup_step = 0
        if self.is_master and self.track_cfg['enabled']:
            import wandb
            if wandb.run is not None:
                wandb_catchup_step = wandb.run.step
                if self.step < wandb_catchup_step:
                    print(f"PyTorch is at step {self.step}, but WandB is at {wandb_catchup_step}.")
                    print("Silencing logs and skipping evaluations until PyTorch catches up...")

        # --- 2. Calculate Dataloader Position ---
        epoch_length = len(self.dataset)
        start_epoch = self.step // epoch_length
        batches_to_skip = self.step % epoch_length

        for epoch in range(start_epoch, self.train_cfg['epochs']):
            # If using DistributedDataParallel, ensure the sampler gets the current epoch seed
            if hasattr(self.dataset, 'sampler') and hasattr(self.dataset.sampler, 'set_epoch'):
                self.dataset.sampler.set_epoch(epoch)

            iterator = tqdm(self.dataset,
                            desc=f"Epoch {epoch + 1}/{self.train_cfg['epochs']}") if self.is_master else self.dataset

            for i, features in enumerate(iterator):
                # --- 3. Fast-Forward Dataloader ---
                # If resuming mid-epoch, skip the batches we already processed before the checkpoint
                if epoch == start_epoch and i < batches_to_skip:
                    continue

                features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                loss = self.train_step(features)

                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')

                # --- 4. Synchronized Logging ---
                if self.is_master:
                    if self.step % self.train_cfg['print_freq'] == 0:
                        # Only log to WandB if we have surpassed the steps from the aborted run
                        if self.step >= wandb_catchup_step:
                            self._write_summary(self.step, features, loss)

                # --- DiffWave Fast Sampling Evaluation & Tracking ---
                if self.step > 0 and self.step % self.train_cfg['checkpoint_freq'] == 0:
                    if self.is_master:
                        self.save_to_checkpoint()

                        # Skip expensive evaluation if WandB already has the data for this step
                        if self.step >= wandb_catchup_step:
                            self.model.eval()
                            self.evaluate()
                            self.model.train()

                if torch.distributed.is_initialized():
                    dist.barrier()

                self.step += 1

    def train_step(self, features):
        for param in self.model.parameters():
            param.grad = None

        signal = features['signal']

        N, T = signal.shape
        device = signal.device
        self.noise_level = self.noise_level.to(device)

        with self.autocast:
            t = torch.randint(0, len(self.noise_level), [N], device=signal.device)
            noise_scale = self.noise_level[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale ** 0.5
            noise = torch.randn_like(signal)
            noisy_signal = noise_scale_sqrt * signal + (1.0 - noise_scale) ** 0.5 * noise

            # --- Capture Real Forward Process (Master Only) ---
            if self.is_master:
                # We track based on the dynamic t picked for the first sample in the batch
                first_sample_t = t[0].item()
                # Pass clean signal, noisy signal, pure noise, current t value, and global step
                self.track_forward_process(signal[0:1], noisy_signal[0:1], noise[0:1], first_sample_t, self.step)

            predicted = self.model(noisy_signal, t)
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
    total_params = config['model']['total_params']

    # 1. Initialize tracking (Master only)
    if learner.is_master:
        is_resuming = config['training'].get('resume', False)

        # Setup WandB ID logic for resuming
        wandb_id_path = os.path.join(run_dir, "wandb_id.txt")
        wandb_run_id = None

        # If resuming, try to fetch the previous WandB ID
        if is_resuming and os.path.exists(wandb_id_path):
            with open(wandb_id_path, "r") as f:
                wandb_run_id = f.read().strip()
            print(f"Found existing WandB run ID: {wandb_run_id}")

        # Turn W&B on
        if config['tracking']['wandb']['enabled']:
            wandb_kwargs = {
                "project": config['experiment']['project_name'],
                "name": os.path.basename(run_dir),
                "config": config
            }

            # Inject the old ID to stitch the runs together
            if wandb_run_id:
                wandb_kwargs["id"] = wandb_run_id
                wandb_kwargs["resume"] = "allow"

            wandb.init(**wandb_kwargs)

            # If starting fresh, save the newly generated ID for the future
            if not wandb_run_id and wandb.run is not None:
                with open(wandb_id_path, "w") as f:
                    f.write(wandb.run.id)

            # Add the parameter count to the wandb config
            wandb.config.update({"total_parameters": total_params})

        # Print to console
        print(f"\n{'=' * 40}")
        print(f"Model Initialized!")
        print(f"Total Trainable Parameters: {total_params:,}")
        print(f"{'=' * 40}\n")

        # 2. Checkpoint restoration or Pre-trained weights initialization
        if is_resuming:
            print("Attempting to resume full training state from checkpoint...")
            learner.restore_from_checkpoint()

        elif config['training'].get('use_pretrained_weights', False):
            weight_path = config['paths'].get('pretrained_weights_path', '')
            print(f"Initializing model with pre-trained weights from: {weight_path}")
            print("Starting with a fresh optimizer from Step 0...")
            learner.load_pretrained_weights(weight_path)

        else:
            print("Starting completely fresh from Step 0 with random weights...")

    # Register the emergency save handler
    register_graceful_shutdown(learner, config)

    # 3. Start training (ALL GPUs must run this)
    learner.train()

    # 4. Finish tracking
    if learner.is_master and config['tracking']['wandb']['enabled']:
        wandb.finish()


def train(config, run_dir):
    dataset = from_path([config['paths']['data_path']], config)
    model = DiffWave(config).cuda()
    _train_impl(0, model, dataset, config, run_dir)


def train_distributed(replica_id, replica_count, port, config, run_dir):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # Increase the timeout to 60 minutes so evaluation has time to finish
    dist.init_process_group(
        'nccl',
        rank=replica_id,
        world_size=replica_count,
        timeout=timedelta(minutes=120)
    )

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

    # Count number of parameters
    temp_model = DiffWave(config)
    total_params = count_parameters(temp_model)
    # Save it to config so we don't have to calculate it again in _train_impl
    config['model']['total_params'] = total_params
    del temp_model  # Free up memory immediatel

    # Auto-generate a unique checkpoint directory based on training AND model params
    layers = config['model']['residual_layers']
    channels = config['model']['residual_channels']
    param_str = format_param_count(total_params)
    run_name = f"diff_{param_str}_L{layers}_C{channels}_bs{train_cfg['batch_size']}_lr{train_cfg['learning_rate']}_ep{train_cfg['epochs']}"
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