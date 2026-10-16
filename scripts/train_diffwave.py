import argparse
import datetime
import os
import sys
from datetime import timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import device_count
from torch.multiprocessing import spawn
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import wandb
import yaml

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference_diffwave import run_reverse_diffusion_loop
from models.diffwave import DiffWave
from utils.create_nda import apply_nda_mutations
from utils.dataset import from_path
from utils.helpers import (
    count_parameters,
    format_param_count,
    generate_run_name,
    get_noise_schedule,
    load_config,
    plot_noise_schedule,
    register_graceful_shutdown,
    set_seed,
    setup_device,
)
from utils.metrics import calculate_f_mem, calculate_fid, calculate_kid


# ==============================================================================
# NDA Batch Helper (Applies utils/create_nda.py mutations)
# ==============================================================================
def apply_nda_batch(x_0_tensor, energy_max=3000.0):
    """
    Applies the exact apply_nda_mutations function from utils/create_nda.py
    to each sample in the PyTorch batch.
    """
    device = x_0_tensor.device
    B, L = x_0_tensor.shape
    x_0_np = x_0_tensor.detach().cpu().numpy()

    y_ood_list = []
    for i in range(B):
        mutated_y, _, _ = apply_nda_mutations(x_0_np[i], energy_max=energy_max)
        y_ood_list.append(mutated_y)

    y_ood_np = np.stack(y_ood_list, axis=0)
    return torch.from_numpy(y_ood_np).to(device=device, dtype=torch.float32)


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


# ==============================================================================
# Core Training Logic
# ==============================================================================
class DiffWaveLearner:

    def __init__(self, model_dir, model, dataset, optimizer, config):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.config = config

        self.device = next(self.model.parameters()).device

        self.train_cfg = config["training"]
        self.model_cfg = config["model"]
        self.hw_cfg = config["hardware"]
        self.track_cfg = config["tracking"]["wandb"]
        self.f_mem_threshold = config["metrics"]["f_mem_threshold"]

        # Hyperparameters for NDA Loss
        self.use_nda = self.train_cfg.get("use_nda", True)
        self.lambda_nda = self.train_cfg.get("lambda_nda", 0.01)
        self.c_const = self.train_cfg.get("nda_c_const", 10.0)

        self.autocast = torch.amp.autocast(
            "cuda", enabled=self.hw_cfg.get("fp16", False)
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.hw_cfg.get("fp16", False)
        )

        self.step = 0
        self.is_master = True
        self.saved_checkpoints = []

        # Tracking Directories Setup
        self.track_dir = os.path.join(model_dir, "tracked_signals")
        self.track_dir_eval = os.path.join(self.track_dir, "eval")
        self.track_dir_train = os.path.join(self.track_dir, "train")

        if self.is_master:
            os.makedirs(self.track_dir_eval, exist_ok=True)
            os.makedirs(self.track_dir_train, exist_ok=True)

        beta, alpha_bar = get_noise_schedule(self.model_cfg)
        plot_noise_schedule(beta, alpha_bar, model_dir)

        self.noise_level = torch.tensor(
            alpha_bar.astype(np.float32), device=self.device
        )
        self.loss_fn = nn.L1Loss()

        self.max_val = self.dataset.dataset.max_val

        if self.is_master:
            scaling_path = self.config["paths"]["scaling_factor_path"]
            os.makedirs(os.path.dirname(scaling_path), exist_ok=True)
            np.save(scaling_path, self.max_val)

    def state_dict(self):
        if hasattr(self.model, "module") and isinstance(
            self.model.module, nn.Module
        ):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            "step": self.step,
            "model": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            },
            "optimizer": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in self.optimizer.state_dict().items()
            },
            "config": self.config,
            "scaler": self.scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, "module") and isinstance(
            self.model.module, nn.Module
        ):
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scaler.load_state_dict(state_dict["scaler"])
        self.step = state_dict["step"]

        if "torch_rng_state" in state_dict:
            torch.set_rng_state(state_dict["torch_rng_state"])
            torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
            np.random.set_state(state_dict["numpy_rng_state"])

    def save_to_checkpoint(self, filename="weights"):
        save_basename = f"{filename}-{self.step}.pt"
        save_name = f"{self.model_dir}/{save_basename}"
        link_name = f"{self.model_dir}/{filename}.pt"
        torch.save(self.state_dict(), save_name)
        if os.name == "nt":
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

            self.saved_checkpoints.append(save_name)
            keep_n = self.train_cfg.get("keep_last_n_checkpoints", 5)

            if len(self.saved_checkpoints) > keep_n:
                oldest_ckpt = self.saved_checkpoints.pop(0)
                if os.path.exists(oldest_ckpt):
                    os.remove(oldest_ckpt)

    def restore_from_checkpoint(self, filename="weights"):
        try:
            checkpoint = torch.load(
                f"{self.model_dir}/{filename}.pt", weights_only=False
            )
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def load_pretrained_weights(self, filepath):
        try:
            checkpoint = torch.load(
                filepath, map_location="cpu", weights_only=False
            )
            model_state = checkpoint.get("model", checkpoint)

            if hasattr(self.model, "module") and isinstance(
                self.model.module, nn.Module
            ):
                self.model.module.load_state_dict(model_state)
            else:
                self.model.load_state_dict(model_state)

            return True
        except Exception as e:
            if self.is_master:
                print(
                    f"Failed to load pre-trained weights from {filepath}. Error: {e}"
                )
            raise

    # ==============================================================================
    # Evaluation Routine
    # ==============================================================================
    def evaluate(self):
        """Generates fast samples and calculates FID/KID metrics using optimized batch sizes."""
        num_samples_to_generate = 256

        # --- 0. Isolate RNG State ---
        torch_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()

        eval_batch_size = 64

        print(
            f"\nGenerating {num_samples_to_generate} samples for evaluation (Eval Batch Size: {eval_batch_size})..."
        )

        model_to_eval = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        model_to_eval.eval()

        # --- 1. Collect Real Samples ---
        all_real_data = []
        real_collected = 0

        for eval_batch in self.dataset:
            real_batch = eval_batch["signal"]
            all_real_data.append(real_batch.cpu().numpy())
            real_collected += real_batch.size(0)

            if real_collected >= num_samples_to_generate:
                break

        real_np = np.concatenate(all_real_data, axis=0)[
            :num_samples_to_generate
        ]

        # --- 2. Generate Fake Samples ---
        all_fake_data = []
        generated_so_far = 0

        with torch.no_grad():
            while generated_so_far < num_samples_to_generate:
                current_batch_size = min(
                    eval_batch_size, num_samples_to_generate - generated_so_far
                )
                current_track_dir = (
                    self.track_dir_eval if generated_so_far == 0 else None
                )

                fake_signal, _ = run_reverse_diffusion_loop(
                    model=model_to_eval,
                    config=self.config,
                    num_samples=current_batch_size,
                    device=self.device,
                    track_dir=current_track_dir,
                    global_step=self.step,
                    max_val=self.max_val,
                )

                all_fake_data.append(fake_signal.cpu().numpy())
                generated_so_far += current_batch_size

        fake_np = np.concatenate(all_fake_data, axis=0)

        # --- 3. Calculate Metrics ---
        try:
            fd_score = calculate_fid(real_np, fake_np)
            kid_score = calculate_kid(real_np, fake_np)
            f_mem_percentage = calculate_f_mem(
                fake_np, real_np, k=self.f_mem_threshold
            )
            if self.track_cfg["enabled"]:
                import wandb

                wandb.log(
                    {
                        "eval/FID": fd_score,
                        "eval/KID": kid_score,
                        "eval/f_mem": f_mem_percentage,
                    },
                    step=self.step,
                )

            print(
                f"--> Eval Step {self.step} | FD: {fd_score:.4f} | KD: {kid_score:.4f} | F_Mem: {f_mem_percentage:.1f}"
            )
        except Exception as e:
            print(
                f"\nWarning: Failed to calculate evaluation metrics. Error: {e}"
            )

        # --- 4. Restore RNG State ---
        torch.set_rng_state(torch_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

    # ==============================================================================
    # Tracking x_0_hat Reconstruction Plot
    # ==============================================================================
    def track_reconstruction(
        self,
        clean_signal,
        noisy_signal,
        x_0_hat,
        t_val,
        global_step,
        calib_params=None,
    ):
        """Plots and saves Ground Truth (x_0), Noisy Input (x_t), and Reconstructed (hat_x_0)."""
        T_total = self.model_cfg["T"]
        interesting_t_values = [0, T_total // 2 - 1, T_total - 1]

        if t_val not in interesting_t_values:
            return

        math_step = t_val + 1

        calib_str = ""
        if calib_params is not None:
            coeffs = (
                calib_params[0].cpu().numpy()
                if hasattr(calib_params, "cpu")
                else calib_params[0]
            )
            calib_str = f" | Calib Coeffs: [{', '.join(f'{x:.4f}' for x in coeffs)}]"

        current_track_path = os.path.join(
            self.track_dir_train, f"x0_hat_step_{math_step}"
        )
        os.makedirs(current_track_path, exist_ok=True)

        clean_np = clean_signal[0].detach().cpu().numpy()
        noisy_np = noisy_signal[0].detach().cpu().numpy()
        pred_x0_np = x_0_hat[0].detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            clean_np,
            label="$x_0$ (Ground Truth)",
            color="blue",
            linewidth=1.2,
            alpha=0.8,
        )
        ax.plot(
            noisy_np,
            label=f"$x_{{{math_step}}}$ (Noisy Input)",
            color="gray",
            linestyle="--",
            alpha=0.5,
        )
        ax.plot(
            pred_x0_np,
            label=r"$\hat{x}_0$ (Reconstructed)",
            color="crimson",
            linewidth=1.2,
        )

        ax.set_title(
            f"Reconstruction ($\hat{{x}}_0$) vs Input ($x_{{{math_step}}}$) | Global Step: {global_step}{calib_str}"
        )
        ax.set_ylabel("Amplitude (Normalized)")
        ax.set_xlabel("Channel / Bin")
        ax.legend(loc="upper right")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                current_track_path, f"x0_hat_step_{global_step:06d}.png"
            ),
            dpi=200,
        )
        plt.close(fig)

    def train_step(self, features):
        for param in self.model.parameters():
            param.grad = None

        full_signal = features["signal"]
        B, full_len = full_signal.shape

        calib_cfg = self.config.get("calibration", {})
        is_calib = calib_cfg.get("enabled", False)
        cond_method = calib_cfg.get("conditioning_method", "concat")

        conditioner = (
            features.get("conditioner", None)
            if (is_calib and cond_method == "film")
            else None
        )

        with self.autocast:
            t = torch.randint(
                0, len(self.noise_level), [B], device=self.device
            )
            noise_scale = self.noise_level[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale**0.5
            one_minus_alpha_sqrt = (1.0 - noise_scale) ** 0.5

            noise = torch.randn_like(full_signal)
            noisy_signal = (
                noise_scale_sqrt * full_signal + one_minus_alpha_sqrt * noise
            )

            # 1. Forward Pass - Predict Noise
            predicted = self.model(noisy_signal, t, conditioner=conditioner)
            pred_noise = predicted.squeeze(1)

            # 2. Reconstruct x_0_hat from predicted noise
            x_0_hat = (
                noisy_signal - one_minus_alpha_sqrt * pred_noise
            ) / (noise_scale_sqrt + 1e-8)

            # Isolate spectrum signal if calibration concatenation is used
            if is_calib and cond_method == "concat":
                spec_clean = full_signal[:, :-4]
                spec_noisy = noisy_signal[:, :-4]
                spec_x0_hat = x_0_hat[:, :-4]
                calib_params_plot = full_signal[0:1, -4:]
            else:
                spec_clean = full_signal
                spec_noisy = noisy_signal
                spec_x0_hat = x_0_hat
                calib_params_plot = None

            # 3. Compute Standard Diffusion Loss
            loss_diffwave = self.loss_fn(noise, pred_noise)

            # 4. Compute OOD/NDA Regularization Loss using utils/create_nda.py
            if self.use_nda:
                y_ood = apply_nda_batch(spec_clean, energy_max=3000.0)
                ood_dist = torch.mean((spec_x0_hat - y_ood) ** 2, dim=-1)
                loss_nda = torch.mean(1.0 / (ood_dist + self.c_const))
                loss = loss_diffwave + self.lambda_nda * loss_nda
            else:
                loss_nda = torch.tensor(0.0, device=self.device)
                loss = loss_diffwave

            # 5. Track plots for master rank
            if self.is_master:
                first_sample_t = t[0].item()
                self.track_reconstruction(
                    spec_clean[0:1],
                    spec_noisy[0:1],
                    spec_x0_hat[0:1],
                    first_sample_t,
                    self.step,
                    calib_params=calib_params_plot,
                )

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.train_cfg.get("max_grad_norm") or 1e9,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.loss_diffwave_val = loss_diffwave.item()
        self.loss_nda_val = loss_nda.item() if self.use_nda else 0.0

        return loss

    def train(self):
        wandb_catchup_step = 0
        if self.is_master and self.track_cfg["enabled"]:
            import wandb

            if wandb.run is not None:
                wandb_catchup_step = wandb.run.step

        epoch_length = len(self.dataset)
        start_epoch = self.step // epoch_length
        batches_to_skip = self.step % epoch_length

        for epoch in range(start_epoch, self.train_cfg["epochs"]):
            if hasattr(self.dataset, "sampler") and hasattr(
                self.dataset.sampler, "set_epoch"
            ):
                self.dataset.sampler.set_epoch(epoch)

            iterator = (
                tqdm(
                    self.dataset,
                    desc=f"Epoch {epoch + 1}/{self.train_cfg['epochs']}",
                )
                if self.is_master
                else self.dataset
            )

            for i, features in enumerate(iterator):
                if epoch == start_epoch and i < batches_to_skip:
                    continue

                features = _nested_map(
                    features,
                    lambda x: (
                        x.to(self.device) if isinstance(x, torch.Tensor) else x
                    ),
                )
                loss = self.train_step(features)

                if torch.isnan(loss).any():
                    raise RuntimeError(
                        f"Detected NaN loss at step {self.step}."
                    )

                if self.is_master:
                    if self.step % self.train_cfg["print_freq"] == 0:
                        if self.step >= wandb_catchup_step:
                            self._write_summary(self.step, features, loss)

                # --- Checkpoint & Evaluation Trigger ---
                if (
                    self.step > 0
                    and self.step % self.train_cfg["checkpoint_freq"] == 0
                ):
                    if self.is_master:
                        self.save_to_checkpoint()

                        if self.step >= wandb_catchup_step:
                            self.model.eval()
                            self.evaluate()
                            self.model.train()

                if torch.distributed.is_initialized():
                    dist.barrier()

                self.step += 1

    def _write_summary(self, step, features, loss):
        fig, ax = plt.subplots()
        ax.plot(features["signal"][0].cpu().numpy())
        ax.set_title(f"DiffWave Training Signal at step {step}")

        current_epoch = step // len(self.dataset)

        if self.track_cfg["enabled"]:
            wandb.log(
                {
                    "train/epoch": current_epoch,
                    "train/global_step": step,
                    "train/total_loss": loss.item(),
                    "train/loss_diffwave": self.loss_diffwave_val,
                    "train/loss_nda": self.loss_nda_val,
                    "train/grad_norm_diffwave": self.grad_norm,
                    "feature/signal_diffwave": wandb.Image(fig),
                },
                step=step,
            )

        plt.close(fig)


# ==============================================================================
# Execution & Distribution Handlers
# ==============================================================================
def _train_impl(replica_id, model, dataset, config, run_dir):
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )

    learner = DiffWaveLearner(run_dir, model, dataset, opt, config)
    learner.is_master = replica_id == 0
    total_params = config["model"]["total_params"]

    if learner.is_master:
        is_resuming = config["training"].get("resume", False)
        wandb_id_path = os.path.join(run_dir, "wandb_id.txt")
        wandb_run_id = None

        if is_resuming and os.path.exists(wandb_id_path):
            with open(wandb_id_path, "r") as f:
                wandb_run_id = f.read().strip()

        if config["tracking"]["wandb"]["enabled"]:
            wandb_kwargs = {
                "project": config["experiment"]["project_name"],
                "name": os.path.basename(run_dir),
                "config": config,
            }

            if wandb_run_id:
                wandb_kwargs["id"] = wandb_run_id
                wandb_kwargs["resume"] = "allow"

            wandb.init(**wandb_kwargs)

            if not wandb_run_id and wandb.run is not None:
                with open(wandb_id_path, "w") as f:
                    f.write(wandb.run.id)

            wandb.config.update({"total_parameters": total_params})

        print(f"\n{'=' * 40}")
        print("Model Initialized!")
        print(f"Total Trainable Parameters: {total_params:,}")
        print(f"{'=' * 40}\n")

        if is_resuming:
            learner.restore_from_checkpoint()
        elif config["training"].get("use_pretrained_weights", False):
            weight_path = config["paths"].get("pretrained_weights_path", "")
            learner.load_pretrained_weights(weight_path)

    register_graceful_shutdown(learner, config)
    learner.train()

    if learner.is_master and config["tracking"]["wandb"]["enabled"]:
        wandb.finish()


def train(config, run_dir):
    device = setup_device(config["hardware"], mode="training")

    base_dir = config["paths"]["base_dir"]
    dataset_name = config["paths"]["dataset_name"]
    dataset_dir = os.path.join(base_dir, dataset_name)
    dataset = from_path(dataset_dir, config)

    model = DiffWave(config).to(device)

    _train_impl(
        replica_id=0,
        model=model,
        dataset=dataset,
        config=config,
        run_dir=run_dir,
    )


def train_distributed(replica_id, replica_count, port, config, run_dir):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "nccl",
        rank=replica_id,
        world_size=replica_count,
        timeout=timedelta(minutes=120),
    )

    dataset = from_path(
        config["paths"]["dataset_dir"], config, is_distributed=True
    )
    device = torch.device("cuda", replica_id)
    torch.cuda.set_device(device)
    model = DiffWave(config).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id])
    _train_impl(replica_id, model, dataset, config, run_dir)


def _get_free_port():
    import socketserver

    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]


def main():
    parser = argparse.ArgumentParser(description="Train 1-D DiffWave Model")
    parser.add_argument(
        "--exp",
        type=str,
        default="configs/experiments/diffwave_base.yaml",
        help="Path to experiment config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="configs/models/diffwave.yaml",
        help="Path to model config",
    )
    args = parser.parse_args()

    config = load_config(args.exp, args.model)
    train_cfg = config["training"]
    hw_cfg = config["hardware"]
    path_cfg = config["paths"]

    base_dir = path_cfg["base_dir"]
    dataset_name = path_cfg["dataset_name"]

    dataset_dir = os.path.join(base_dir, dataset_name)
    config["paths"]["dataset_dir"] = dataset_dir
    config["paths"]["data_path"] = os.path.join(
        dataset_dir, "spectrum_matrix.npy"
    )
    config["paths"]["scaling_factor_path"] = os.path.join(
        dataset_dir, "scaling_factor.npy"
    )

    calib_cfg = config.get("calibration", {})
    is_calib = calib_cfg.get("enabled", False)
    num_calib_params = calib_cfg.get("num_params", 4)
    conditioning_method = calib_cfg.get("conditioning_method", "concat")

    if is_calib and conditioning_method == "concat":
        config["model"]["signal_len"] += num_calib_params

    set_seed(train_cfg["seed"])

    temp_model = DiffWave(config)
    total_params = count_parameters(temp_model)
    config["model"]["total_params"] = total_params
    del temp_model

    run_name = generate_run_name(config, total_params)
    run_dir = os.path.join(path_cfg["checkpoints_dir"], run_name)

    replica_count = (
        hw_cfg["num_gpus"] if hw_cfg["num_gpus"] > 0 else device_count()
    )

    if replica_count > 1:
        if train_cfg["batch_size"] % replica_count != 0:
            raise ValueError(
                f"Batch size {train_cfg['batch_size']} is not evenly divisible by # GPUs {replica_count}."
            )
        config["training"]["batch_size"] = (
            train_cfg["batch_size"] // replica_count
        )
        port = _get_free_port()
        spawn(
            train_distributed,
            args=(replica_count, port, config, run_dir),
            nprocs=replica_count,
            join=True,
        )
    else:
        train(config, run_dir)


if __name__ == "__main__":
    main()