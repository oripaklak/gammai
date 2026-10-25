import os
import sys
import time
import datetime
import argparse
import torch
import torch.optim as optim
import numpy as np
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Add the project root to the Python path to import from models/ and utils/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.helpers as helpers
from utils.metrics import calculate_fid, calculate_kid
from utils.dataset import from_path
from utils.checkpoints import CheckpointManager
from utils.plot import plot_and_save_spectra
from models.vae import SpectrumVAE, vae_loss_function


# ==============================================================================
# Core Training & Evaluation Logic
# ==============================================================================

def evaluate(model, dataloader, device, latent_dim, current_epoch, eval_track_dir, max_val,
             loss_fn_vae=None, beta=1.0):
    """
    Evaluates both:
      1. Reconstruction Fidelity (Crucial for VAE as a compressor for DiffWave)
      2. Prior Sampling Fidelity (FID / KID from z ~ N(0, I))
      3. Validation Loss and Reconstruction Overlay Plotting
    """
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    all_real_data = []
    all_fake_data = []

    with torch.no_grad():
        for val_batch in dataloader:
            real_batch = val_batch["signal"].to(device)
            B = real_batch.size(0)

            # Sample from latent prior: z ~ N(0, I)
            z = torch.randn(B, latent_dim, device=device)
            generated_batch = raw_model.decode(z)

            if generated_batch.dim() == 3:
                generated_batch = generated_batch.squeeze(1)

            all_real_data.append(real_batch.detach().cpu().numpy())
            all_fake_data.append(generated_batch.detach().cpu().numpy())

    real_np = np.concatenate(all_real_data, axis=0)
    fake_np = np.concatenate(all_fake_data, axis=0)

    metrics = {}

    # 1. Metrics Calculation
    try:
        fd_score = calculate_fid(real_np, fake_np)
        kid_score = calculate_kid(real_np, fake_np)
        metrics["eval/FID"] = fd_score
        metrics["eval/KID"] = kid_score
        print(f"--> Eval Epoch {current_epoch} | FD: {fd_score:.4f} | KD: {kid_score:.4f}")
    except Exception as e:
        print(f"Warning: Failed to calculate evaluation metrics. Error: {e}")

    # 2. Plotting using utils/helpers.py
    if eval_track_dir is not None:
        epoch_save_dir = os.path.join(eval_track_dir, f"epoch_{current_epoch:06d}")
        os.makedirs(epoch_save_dir, exist_ok=True)

        # Unnormalize to raw integer counts if max_val is provided
        if max_val is not None:
            fake_counts = helpers.unnormalize(fake_np[:3], max_val)
            fake_counts = np.clip(np.round(fake_counts), a_min=0, a_max=None)
        else:
            fake_counts = fake_np[:3]

        # Call the helper function
        plot_and_save_spectra(
            generated_counts=fake_counts,
            save_dir=epoch_save_dir,
            model_name=f"VAE_Epoch_{current_epoch}",
            detect_peaks=False,
        )

    return metrics

def train(model, dataloader, optimizer, device, config, checkpoint_dir, run_name):
    """Executes the main training loop with unified checkpoint management."""
    train_cfg = config["training"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    track_cfg = config["tracking"]["wandb"]

    # Derive tracked_signals/eval directory: experiments/<run_name>/tracked_signals/eval
    run_dir = os.path.dirname(checkpoint_dir)
    eval_track_dir = os.path.join(run_dir, "tracked_signals", "eval")
    os.makedirs(eval_track_dir, exist_ok=True)

    max_val = getattr(dataloader.dataset, "max_val", None)

    print(f"Starting training for {train_cfg['epochs']} epochs...")
    global_step = 0

    # Initialize CheckpointManager
    ckpt_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizer=optimizer,
        config=config,
        keep_last_n=train_cfg.get("keep_last_n_checkpoints", 5),
        is_master=True,
    )

    for epoch in range(train_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        total_mse_loss = 0.0

        for batch in dataloader:
            batch_data = batch["signal"].to(device)

            optimizer.zero_grad()
            reconstructed_batch, mu, logvar = model(batch_data)
            batch_mse, loss = vae_loss_function(
                reconstructed_batch, batch_data, mu, logvar, loss_cfg["beta"]
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=train_cfg.get("max_grad_norm", 5.0)
            )

            train_loss += loss.item()
            total_mse_loss += batch_mse.item()

            optimizer.step()
            global_step += 1

        avg_train_loss = train_loss / len(dataloader.dataset)
        avg_mse_loss = total_mse_loss / len(dataloader.dataset)
        current_epoch = epoch + 1

        grad_norm_val = (
            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        )

        metrics = {
            "train/epoch": current_epoch,
            "train/loss_vae": avg_train_loss,
            "train/mse_loss": avg_mse_loss,
            "train/global_step": global_step,
            "train/grad_norm_vae": grad_norm_val,
        }

        # Evaluation Pipeline Call with eval_track_dir and max_val
        if current_epoch % train_cfg.get("eval_freq", 20) == 0:
            eval_metrics = evaluate(
                model=model,
                dataloader=dataloader,
                device=device,
                latent_dim=model_cfg["latent_dim"],
                current_epoch=current_epoch,
                eval_track_dir=eval_track_dir,
                max_val=max_val,
                loss_fn_vae=vae_loss_function,
                beta=loss_cfg.get("beta", 1.0),
            )
            metrics.update(eval_metrics)
            model.train()

        if track_cfg.get("enabled", True):
            wandb.log(metrics)

        if current_epoch % train_cfg.get("print_freq", 10) == 0:
            print(
                f"Epoch [{current_epoch}/{train_cfg['epochs']}] | "
                f"Train Loss: {avg_train_loss:.4f} | MSE Loss: {avg_mse_loss:.6f} | "
                f"Grad Norm: {grad_norm_val:.2f}"
            )

        if current_epoch % train_cfg.get("checkpoint_freq", 1000) == 0:
            saved_path = ckpt_manager.save(
                step_or_epoch=current_epoch, filename="weights"
            )
            print(f"--- Saved intermediate checkpoint: {saved_path} ---")

    print("Waiting 5 seconds for file I/O to finish...")
    time.sleep(5)

# ==============================================================================
# Initialization & Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Spectrum VAE")
    parser.add_argument(
        "--exp",
        type=str,
        default="configs/experiments/vae_base.yaml",
        help="Path to the experiment config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="configs/models/vae.yaml",
        help="Path to the model config",
    )
    args = parser.parse_args()

    # 1. Load Configurations & Set Seed
    config = helpers.load_config(args.exp, args.model)
    train_cfg = config["training"]
    model_cfg = config["model"]
    path_cfg = config["paths"]
    track_cfg = config["tracking"]["wandb"]

    helpers.set_seed(train_cfg["seed"])
    
    base_dir = path_cfg["base_dir"]
    dataset_name = path_cfg["dataset_name"]

    dataset_dir = os.path.join(base_dir, dataset_name)
    path_cfg["dataset_dir"] = dataset_dir
    path_cfg["data_path"] = os.path.join(dataset_dir, "spectrum_matrix.npy")
    path_cfg["scaling_factor_path"] = os.path.join(dataset_dir, "scaling_factor.npy")
    
    # 2. Load and Normalize Data (Derive Signal Length L)
    print(f"Loading data from {path_cfg['data_path']}...")
    dataloader = from_path([path_cfg["data_path"]], config)
    dataset_instance = dataloader.dataset
    N, L = dataset_instance.matrix.shape

    os.makedirs(os.path.dirname(path_cfg["scaling_factor_path"]), exist_ok=True)
    np.save(path_cfg["scaling_factor_path"], dataset_instance.max_val)

    # 3. Initialize Device & Model (Single Instantiation)
    device = helpers.setup_device(config["hardware"])
    model = SpectrumVAE(L=L, latent_dim=model_cfg["latent_dim"]).to(device)

    # 4. Count Parameters & Build run_name

    # Clean float format for learning rate (e.g., 0.0002 -> 2e-4)
    lr_str = f"{lr:.0e}".replace("-0", "-") if isinstance(lr, float) else lr
    # Format training duration (step-based with fallback to epochs if still epoch-based)
    if steps is not None:
        steps_str = (
            f"{steps // 1000}k"
            if isinstance(steps, int) and steps >= 1000
            else f"{steps}"
        )
        duration_str = f"steps{steps_str}"
    else:
        duration_str = f"ep{train_cfg.get('epochs', 'unk')}"
    total_params = helpers.count_parameters(model)
    param_str = helpers.format_param_count(total_params)
    print(f"Model Initialized! Total Trainable Parameters: {total_params:,} ({param_str})")

    # 2. Assemble standardized run name:
    # [Model]_[Dataset]_[Arch]_[TrainingParams]_[Seed]
    run_name = (
        f"vae_{param_str}_"
        f"{path_cfg['dataset_name']}"
        f"ld{model_cfg['latent_dim']}_"
        f"bs{train_cfg['batch_size']}_"
        f"lr{lr_str}_"
        f"{duration_str}_"
        f"s{seed}"
    )

    # Setup Experiment Directory & Checkpoint Directory
    run_dir = os.path.join(path_cfg["checkpoints_dir"], run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 5. Initialize W&B (with run ID tracking & resume support)
    if track_cfg.get("enabled", True):
        is_resuming = train_cfg.get("resume", False)
        wandb_id_path = os.path.join(run_dir, "wandb_id.txt")
        wandb_run_id = None

        # Check if an existing run ID is available to resume
        if is_resuming and os.path.exists(wandb_id_path):
            with open(wandb_id_path, "r") as f:
                wandb_run_id = f.read().strip()

        wandb_kwargs = {
            "project": config["experiment"]["project_name"],
            "name": run_name,
            "config": {
                **config,
                "num_trainable_parameters": total_params,
            },
        }

        # Attach ID and set resume mode if resuming
        if wandb_run_id:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "allow"

        wandb.init(**wandb_kwargs)

        # Save the run ID locally so future runs can resume seamlessly
        if not wandb_run_id and wandb.run is not None:
            with open(wandb_id_path, "w") as f:
                f.write(wandb.run.id)

    # 6. Multi-GPU Setup & Optimizer
    if config["hardware"].get("num_gpus", 1) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    if track_cfg.get("enabled", True) and track_cfg.get("log_model_graph", True):
        wandb.watch(model, log="all", log_freq=track_cfg.get("watch_log_freq", 10))

    # 7. Launch Training
    train(model, dataloader, optimizer, device, config, checkpoint_dir, run_name)


if __name__ == "__main__":
    main()