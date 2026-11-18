import os
import ssl

# Bypass SSL inspection issues
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
from datetime import timedelta
import os
import sys

import numpy as np
import torch
from torch.cuda import device_count
import torch.distributed as dist
from torch.multiprocessing import spawn
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import wandb

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference_diffwave import run_reverse_diffusion_loop
from models.compressor import create_compressor
from models.diffwave import DiffWave
from models.loss import LimitedDataLoss
from utils.checkpoints import CheckpointManager
from utils.create_nda import apply_nda_mutations
from utils.dataset import from_path
import utils.helpers as helpers
from utils.metrics import calculate_f_mem, calculate_fid, calculate_kid
from utils.plot import plot_noise_schedule


def apply_nda_batch(x_0_tensor):
    device = x_0_tensor.device
    B, L = x_0_tensor.shape
    x_0_np = x_0_tensor.detach().cpu().numpy()

    y_ood_list = []
    for i in range(B):
        mutated_y, _, _ = apply_nda_mutations(x_0_np[i])
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

    def __init__(self, model_dir, model, dataset, optimizer, config, compressor):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.ckpt_dir = os.path.join(self.model_dir, "checkpoints")
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.config = config
        self.compressor = compressor

        self.device = next(self.model.parameters()).device

        self.train_cfg = config["training"]
        self.eval_cfg = config.get("evaluation", {})
        self.model_cfg = config["model"]
        self.hw_cfg = config["hardware"]
        self.track_cfg = config["tracking"]["wandb"]
        self.loss_cfg = config.get("loss", {})
        self.vae_cfg = config.get("vae", {})
        self.vae_enabled = self.vae_cfg.get("enabled", False)
        self.f_mem_threshold = config["metrics"]["f_mem_threshold"]

        # Strategy Switch
        # Rule 1: vae = True  -> use_nda = True, LimitedDataLoss
        # Rule 2: vae = False -> use_nda = True/False, L1/L2 noise comparison
        self.lambda_nda = self.loss_cfg.get("lambda_nda", 0.01)
        self.eta = self.loss_cfg.get("eta", 100.0)

        if self.vae_enabled:
            self.use_nda = True
            self.criterion = LimitedDataLoss(eta=self.eta)
            self.mode = "latent_limited_data"
        else:
            self.use_nda = self.loss_cfg.get("use_nda", False)
            loss_type = self.loss_cfg.get("diffusion_loss_type", "l1").lower()
            self.noise_criterion = nn.L1Loss() if loss_type == "l1" else nn.MSELoss()
            self.mode = "raw_signal"

        self.autocast = torch.amp.autocast("cuda", enabled=self.hw_cfg.get("fp16", False))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.hw_cfg.get("fp16", False))

        self.step = 0
        self.is_master = True

        self.track_dir = os.path.join(model_dir, "tracked_signals")
        self.track_dir_eval = os.path.join(self.track_dir, "eval")
        self.track_dir_train = os.path.join(self.track_dir, "train")

        if self.is_master:
            os.makedirs(self.track_dir_eval, exist_ok=True)
            os.makedirs(self.track_dir_train, exist_ok=True)

        beta, alpha_bar = helpers.get_noise_schedule(self.model_cfg)
        plot_noise_schedule(beta, alpha_bar, model_dir)

        self.noise_level = torch.tensor(alpha_bar.astype(np.float32), device=self.device)
        self.max_val = self.dataset.dataset.max_val

        if self.is_master:
            scaling_path = self.config["paths"]["scaling_factor_path"]
            os.makedirs(os.path.dirname(scaling_path), exist_ok=True)
            np.save(scaling_path, self.max_val)

        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=self.ckpt_dir,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            config=self.config,
            keep_last_n=self.train_cfg.get("keep_last_n_checkpoints", 5),
            is_master=self.is_master,
        )

    def state_dict(self):
        return self.ckpt_manager.get_state_dict(step_or_epoch=self.step)

    def load_state_dict(self, state_dict):
        self.step = self.ckpt_manager.load_state_dict(state_dict)

    def save_to_checkpoint(self, filename="weights"):
        self.ckpt_manager.save(step_or_epoch=self.step, filename=filename)

    def restore_from_checkpoint(self, filename="weights"):
        success, step = self.ckpt_manager.restore(filename=filename)
        if success:
            self.step = step
        return success

    def load_pretrained_weights(self, filepath):
        return self.ckpt_manager.load_pretrained(filepath)

    def evaluate(self):
        """Generates fast samples, decodes if latent, and evaluates FID/KID."""
        num_samples_to_generate = self.eval_cfg.get("num_samples_to_generate", 256)
        eval_batch_size = self.eval_cfg.get("eval_batch_size", 64)

        torch_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()

        print(f"\nGenerating {num_samples_to_generate} samples for evaluation...")

        # Safe unwrapping of DataParallel / DDP
        model_to_eval = self.model.module if hasattr(self.model, "module") else self.model
        model_to_eval.eval()

        # Collect real samples in raw signal space
        all_real_data = []
        real_collected = 0
        for eval_batch in self.dataset:
            real_batch = eval_batch["signal"]
            all_real_data.append(real_batch.cpu().numpy())
            real_collected += real_batch.size(0)
            if real_collected >= num_samples_to_generate:
                break
        real_np = np.concatenate(all_real_data, axis=0)[:num_samples_to_generate]

        # Reverse diffusion generation
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
                    compressor=self.compressor if self.vae_enabled else None,
                )

                all_fake_data.append(fake_signal.cpu().numpy())
                generated_so_far += current_batch_size

        fake_np = np.concatenate(all_fake_data, axis=0)

        try:
            fd_score = calculate_fid(real_np, fake_np)
            kid_score = calculate_kid(real_np, fake_np)
            f_mem_percentage = calculate_f_mem(
                fake_np, real_np, k=self.f_mem_threshold
            )
            if self.track_cfg["enabled"]:
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
            print(f"\nWarning: Failed to calculate evaluation metrics: {e}")

        torch.set_rng_state(torch_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

    # def track_train_step(self, spec_raw, pred_clean, conditioner=None):
    #     """Saves a plot of the current training batch clean prediction."""
    #     step_dir = os.path.join(self.track_dir_train, f"step_{self.step}")
    #     os.makedirs(step_dir, exist_ok=True)
    # 
    #     with torch.no_grad():
    #         if self.vae_enabled:
    #             pred_latent = pred_clean[0:1, :-4] if (
    #                         self.config.get("calibration", {}).get("conditioning_method") == "concat") else pred_clean[
    #                                                                                                         0:1]
    #             pred_signal = self.compressor.decode(pred_latent)[0]
    #         else:
    #             pred_signal = pred_clean[0, :-4] if (
    #                         self.config.get("calibration", {}).get("conditioning_method") == "concat") else pred_clean[
    #                 0]
    # 
    #         from utils.plot import plot_diffusion_step
    #         calib_part = conditioner[0] if conditioner is not None else None
    #         bounds_path = os.path.join(self.config["paths"]["dataset_dir"], "global_calib_bounds.json")
    # 
    #         plot_diffusion_step(
    #             track_path=step_dir,
    #             single_signal=pred_signal,
    #             noise_scale_val=0.0,
    #             loop_idx=0,
    #             label=f"train_pred_step_{self.step}",
    #             max_val=self.max_val,
    #             calib_params=calib_part,
    #             bounds_path=bounds_path,
    #         )

    def train_step(self, features):
        for param in self.model.parameters():
            param.grad = None

        full_signal = features["signal"]
        B = full_signal.shape[0]

        calib_cfg = self.config.get("calibration", {})
        is_calib = calib_cfg.get("enabled", False)
        cond_method = calib_cfg.get("conditioning_method", "concat")

        conditioner = (
            features.get("conditioner", None)
            if (is_calib and cond_method == "film")
            else None
        )

        with self.autocast:
            # Separate spectrum from concatenated calibration parameters
            if is_calib and cond_method == "concat":
                spec_raw = full_signal[:, :-4]
                calib_params = full_signal[:, -4:]
                spec_target = self.compressor.encode(spec_raw)
                target = torch.cat([spec_target, calib_params], dim=-1)
            else:
                spec_raw = full_signal
                target = self.compressor.encode(spec_raw)

            # Forward Diffusion
            t = torch.randint(0, len(self.noise_level), [B], device=self.device)
            noise_scale = self.noise_level[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale**0.5
            one_minus_alpha_sqrt = (1.0 - noise_scale) ** 0.5

            noise = torch.randn_like(target)
            noisy_input = noise_scale_sqrt * target + one_minus_alpha_sqrt * noise

            # Predict Noise
            predicted = self.model(noisy_input, t, conditioner=conditioner)
            pred_noise = predicted.squeeze(1)

            # Mode 1: Latent Space + LimitedDataLoss (Case 2)
            if self.mode == "latent_limited_data":
                pred_clean = (noisy_input - one_minus_alpha_sqrt * pred_noise) / (
                    noise_scale_sqrt + 1e-8
                )

                # Mutate raw spectrum and project to latent space
                raw_ood = apply_nda_batch(spec_raw)
                target_ood = self.compressor.encode(raw_ood)

                loss, recon_loss, ood_loss = self.criterion(
                    pred_clean=pred_clean,
                    target_clean=target,
                    ood_sample=target_ood,
                    use_nda=True,
                )
                loss = recon_loss + self.lambda_nda * ood_loss
                self.loss_diffwave_val = recon_loss.item()
                self.loss_nda_val = ood_loss.item()

            # Mode 2: Raw Signal Space + L1/L2 (+ optional NDA)
            else:
                loss_noise = self.noise_criterion(pred_noise, noise)

                if self.use_nda:
                    pred_clean = (noisy_input - one_minus_alpha_sqrt * pred_noise) / (
                        noise_scale_sqrt + 1e-8
                    )
                    ood_sample = apply_nda_batch(spec_raw)
                    diff_ood = (pred_clean - ood_sample).reshape(B, -1)
                    dist_to_ood_sq = torch.sum(diff_ood**2, dim=-1)
                    loss_ood = (1.0 / (dist_to_ood_sq + self.eta)).mean()

                    loss = loss_noise + self.lambda_nda * loss_ood
                    self.loss_nda_val = loss_ood.item()
                else:
                    loss = loss_noise
                    self.loss_nda_val = 0.0

                self.loss_diffwave_val = loss_noise.item()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.train_cfg.get("max_grad_norm", 1.0)
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

    def train(self):
        wandb_catchup_step = 0
        if self.is_master and self.track_cfg["enabled"] and wandb.run is not None:
            wandb_catchup_step = wandb.run.step

        max_steps = self.train_cfg["max_steps"]
        checkpoint_freq = self.train_cfg["checkpoint_freq"]
        print_freq = self.train_cfg["print_freq"]
        eval_freq = self.eval_cfg.get("eval_freq", 2000)

        while True:
            epoch = self.step // len(self.dataset)
            if hasattr(self.dataset, "sampler") and hasattr(self.dataset.sampler, "set_epoch"):
                self.dataset.sampler.set_epoch(epoch)

            iterator = (
                tqdm(self.dataset, desc=f"Epoch {epoch + 1}")
                if self.is_master
                else self.dataset
            )

            for features in iterator:
                if self.step >= max_steps:
                    if self.is_master:
                        print(f"Reached max_steps ({max_steps}). Saving final checkpoint...")
                        self.save_to_checkpoint()
                    if torch.distributed.is_initialized():
                        dist.barrier()
                    return

                features = _nested_map(
                    features,
                    lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x,
                )
                loss = self.train_step(features)

                if torch.isnan(loss).any():
                    raise RuntimeError(f"Detected NaN loss at step {self.step}.")

                self.step += 1

                if self.is_master and self.step % print_freq == 0 and self.step >= wandb_catchup_step:
                    self._write_summary(self.step, loss)

                if self.step > 0 and self.step % checkpoint_freq == 0:
                    if self.is_master:
                        self.save_to_checkpoint()
                        # self.track_train_step(spec_raw, pred_clean, conditioner)
                    if torch.distributed.is_initialized():
                        dist.barrier()

                if self.step % eval_freq == 0:
                    if self.is_master and self.step >= wandb_catchup_step:
                        raw_model = self.model.module if hasattr(self.model, "module") else self.model
                        raw_model.eval()
                        self.evaluate()
                        raw_model.train()
                    if torch.distributed.is_initialized():
                        dist.barrier()

    def _write_summary(self, step, loss):
        current_epoch = step // len(self.dataset)
        if self.track_cfg["enabled"]:
            wandb.log(
                {
                    "train/epoch": current_epoch,
                    "train/global_step": step,
                    "train/total_loss": loss.item(),
                    "train/loss_diffwave": self.loss_diffwave_val,
                    "train/loss_nda": self.loss_nda_val,
                    "train/grad_norm_diffwave": (
                        self.grad_norm.item()
                        if isinstance(self.grad_norm, torch.Tensor)
                        else self.grad_norm
                    ),
                },
                step=step,
            )


# ==============================================================================
# Factory & Execution Handlers
# ==============================================================================
def _train_impl(replica_id, model, dataset, config, run_dir, compressor):
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    learner = DiffWaveLearner(run_dir, model, dataset, opt, config, compressor)
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
        print(f"Operational Mode: {learner.mode.upper()} (Compressor: {compressor.__class__.__name__})")
        print(f"{'=' * 40}\n")

        if is_resuming:
            learner.restore_from_checkpoint()
        elif config["training"].get("use_pretrained_weights", False):
            weight_path = config["paths"].get("pretrained_weights_path", "")
            learner.load_pretrained_weights(weight_path)

    helpers.register_graceful_shutdown(learner, config)
    learner.train()

    if learner.is_master and config["tracking"]["wandb"]["enabled"]:
        wandb.finish()


def train(config, run_dir):
    device = helpers.setup_device(config["hardware"], mode="training")
    dataset_dir = config["paths"]["dataset_dir"]
    dataset = from_path(dataset_dir, config)

    compressor = create_compressor(config, device)
    model = DiffWave(config).to(device)

    _train_impl(0, model, dataset, config, run_dir, compressor)


def train_distributed(replica_id, replica_count, port, config, run_dir):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "nccl",
        rank=replica_id,
        world_size=replica_count,
        timeout=timedelta(minutes=120),
    )

    device = torch.device("cuda", replica_id)
    torch.cuda.set_device(device)

    dataset = from_path(config["paths"]["dataset_dir"], config, is_distributed=True)
    compressor = create_compressor(config, device)

    model = DiffWave(config).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id])

    _train_impl(replica_id, model, dataset, config, run_dir, compressor)
    dist.destroy_process_group()


def _get_free_port():
    import socketserver
    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]


def main():
    parser = argparse.ArgumentParser(description="Train 1-D DiffWave Model")
    parser.add_argument("--exp", type=str, default="configs/experiments/diffwave_base.yaml")
    parser.add_argument("--model", type=str, default="configs/models/diffwave.yaml")
    args = parser.parse_args()

    config = helpers.load_config(args.exp, args.model)
    train_cfg = config["training"]
    hw_cfg = config["hardware"]
    path_cfg = config["paths"]

    base_dir = path_cfg["base_dir"]
    dataset_name = path_cfg["dataset_name"]
    dataset_dir = os.path.join(base_dir, dataset_name)
    path_cfg["dataset_dir"] = dataset_dir
    path_cfg["data_path"] = os.path.join(dataset_dir, "spectrum_matrix.npy")
    path_cfg["scaling_factor_path"] = os.path.join(dataset_dir, "scaling_factor.npy")

    # Record raw length
    temp_data = np.load(path_cfg["data_path"], mmap_mode="r")
    raw_signal_len = temp_data.shape[1]
    path_cfg["raw_signal_len"] = raw_signal_len
    del temp_data

    # Set DiffWave internal signal_len
    vae_cfg = config.get("vae", {})
    calib_cfg = config.get("calibration", {})
    is_calib = calib_cfg.get("enabled", False)
    cond_method = calib_cfg.get("conditioning_method", "concat")
    num_calib_params = calib_cfg.get("num_params", 4)

    if vae_cfg.get("enabled", False):
        config["model"]["signal_len"] = vae_cfg.get("latent_dim", 2048)
    else:
        config["model"]["signal_len"] = raw_signal_len
        if is_calib and cond_method == "concat":
            config["model"]["signal_len"] += num_calib_params

    helpers.set_seed(train_cfg["seed"])

    temp_model = DiffWave(config)
    total_params = helpers.count_parameters(temp_model)
    config["model"]["total_params"] = total_params
    del temp_model

    run_name = helpers.generate_run_name(config, total_params)
    run_dir = os.path.join(path_cfg["checkpoints_dir"], run_name)

    replica_count = hw_cfg["num_gpus"] if hw_cfg["num_gpus"] > 0 else device_count()

    if replica_count > 1:
        if train_cfg["batch_size"] % replica_count != 0:
            raise ValueError(
                f"Batch size {train_cfg['batch_size']} not divisible by GPUs {replica_count}."
            )
        config["training"]["batch_size"] = train_cfg["batch_size"] // replica_count
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