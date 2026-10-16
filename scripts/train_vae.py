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

from utils.helpers import load_config, setup_device, set_seed, count_parameters
from utils.metrics import calculate_frechet_distance, calculate_polynomial_mmd
from utils.dataset import from_path
from models.vae import SpectrumVAE, vae_loss_function


# ==============================================================================
# Core Training & Evaluation Logic
# ==============================================================================

def evaluate(model, dataloader, device, latent_dim, current_epoch):
    """Generates samples, calculates FID/KID metrics, and logs images."""
    model.eval()
    all_real_data = []
    all_fake_data = []

    with torch.no_grad():
        for val_batch in dataloader:
            real_batch = val_batch['signal'].to(device)

            # Generate data from random latent space
            z = torch.randn(real_batch.size(0), latent_dim).to(device)
            generated_batch = model.decode(z).squeeze(1)

            all_real_data.append(real_batch.cpu().numpy())
            all_fake_data.append(generated_batch.cpu().numpy())

    real_np = np.concatenate(all_real_data, axis=0)
    fake_np = np.concatenate(all_fake_data, axis=0)

    metrics = {}
    try:
        fd_score = calculate_frechet_distance(real_np, fake_np)
        kid_score = calculate_polynomial_mmd(real_np, fake_np)

        metrics["eval/FID"] = fd_score
        metrics["eval/KID"] = kid_score

        # Add Image Logging
        fig, ax = plt.subplots()
        ax.plot(fake_np[0])
        ax.set_title(f'VAE Generated Signal (Epoch {current_epoch})')
        metrics["feature/signal_vae"] = wandb.Image(fig)
        plt.close(fig)

        print(f"--> Eval Epoch {current_epoch} | FD: {fd_score:.4f} | KD: {kid_score:.4f}")
    except Exception as e:
        print(f"Warning: Failed to calculate evaluation metrics. Error: {e}")

    return metrics


def save_checkpoint(model, epoch, checkpoint_dir, saved_checkpoints, keep_n):
    """Saves the current model state and removes the oldest checkpoint to save space."""
    ckpt_name = f"checkpoint_ep{epoch}.pth"
    ckpt_save_path = os.path.join(checkpoint_dir, ckpt_name)

    # Unwrap model state_dict safely if DataParallel was used
    state_dict_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state_dict_to_save, ckpt_save_path)
    print(f"--- Saved intermediate checkpoint: {ckpt_save_path} ---")

    # Rolling Checkpoint Logic
    saved_checkpoints.append(ckpt_save_path)

    if len(saved_checkpoints) > keep_n:
        oldest_ckpt = saved_checkpoints.pop(0)
        if os.path.exists(oldest_ckpt):
            os.remove(oldest_ckpt)
            print(f"--- Deleted old checkpoint to save space: {oldest_ckpt} ---")


def train(model, dataloader, optimizer, device, config, checkpoint_dir, run_name):
    """Executes the main training loop."""
    train_cfg = config["training"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    track_cfg = config["tracking"]["wandb"]

    print(f"Starting training for {train_cfg['epochs']} epochs...")
    global_step = 0
    saved_checkpoints = []

    for epoch in range(train_cfg["epochs"]):
        model.train()
        train_loss = 0

        for batch in dataloader:
            batch_data = batch['signal'].to(device)

            optimizer.zero_grad()
            reconstructed_batch, mu, logvar = model(batch_data)
            loss = vae_loss_function(reconstructed_batch, batch_data, mu, logvar, loss_cfg["beta"])

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)

            train_loss += loss.item()
            optimizer.step()
            global_step += 1

        avg_train_loss = train_loss / len(dataloader.dataset)
        current_epoch = epoch + 1

        # Standardized Epoch Metrics
        metrics = {
            "train/epoch": current_epoch,
            "train/loss_vae": avg_train_loss,
            "train/global_step": global_step,
            "train/grad_norm_vae": grad_norm
        }

        # Evaluation Pipeline
        if current_epoch % train_cfg.get("eval_freq", 20) == 0:
            eval_metrics = evaluate(model, dataloader, device, model_cfg["latent_dim"], current_epoch)
            metrics.update(eval_metrics)  # Merge eval metrics into main metrics dictionary
            model.train()  # Safely reset back to train mode

        if track_cfg.get("enabled", True):
            wandb.log(metrics)

        # Print to console
        if current_epoch % train_cfg.get("print_freq", 10) == 0:
            print(f"Epoch [{current_epoch}/{train_cfg['epochs']}] | Train Loss: {avg_train_loss:.4f}")

        # Checkpointing
        if current_epoch % train_cfg.get("checkpoint_freq", 1000) == 0:
            keep_n = train_cfg.get("keep_last_n_checkpoints", 5)
            save_checkpoint(model, current_epoch, checkpoint_dir, saved_checkpoints, keep_n)

    # Save final model
    final_save_path = os.path.join(checkpoint_dir, f"{run_name}_final.pth")
    state_dict_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state_dict_to_save, final_save_path)

    print(f"Training complete. Final model saved locally to: {final_save_path}")
    print("Waiting 5 seconds for file I/O to finish...")
    time.sleep(5)


# ==============================================================================
# Initialization & Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Spectrum VAE")
    parser.add_argument("--exp", type=str, default="configs/experiments/vae_base.yaml",
                        help="Path to the experiment config")
    parser.add_argument("--model", type=str, default="configs/models/vae.yaml", help="Path to the model config")
    args = parser.parse_args()

    # Load configurations
    config = load_config(args.exp, args.model)
    train_cfg = config["training"]
    model_cfg = config["model"]
    path_cfg = config["paths"]
    track_cfg = config["tracking"]["wandb"]

    set_seed(train_cfg["seed"])
    run_name = f"vae_ld{model_cfg['latent_dim']}_bs{train_cfg['batch_size']}_lr{train_cfg['learning_rate']}_ep{train_cfg['epochs']}"

    # Initialize W&B
    if track_cfg.get("enabled", True):
        wandb.init(project=config["experiment"]["project_name"], name=run_name, config=config)

    # Load and Normalize Data
    print(f"Loading data from {path_cfg['data_path']}...")
    dataloader = from_path([path_cfg["data_path"]], config)
    dataset_instance = dataloader.dataset
    N, L = dataset_instance.matrix.shape

    os.makedirs(os.path.dirname(path_cfg["scaling_factor_path"]), exist_ok=True)
    np.save(path_cfg["scaling_factor_path"], dataset_instance.max_val)

    # Initialize Device and Model
    device = setup_device(config["hardware"])
    model = SpectrumVAE(L=L, latent_dim=model_cfg["latent_dim"]).to(device)

    print(f"Total trainable parameters: {count_parameters(model):,}")

    if config["hardware"].get("num_gpus", 1) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    if track_cfg.get("enabled", True) and track_cfg.get("log_model_graph", True):
        wandb.watch(model, log="all", log_freq=track_cfg.get("watch_log_freq", 10))

    # Setup Checkpoint Directory
    run_checkpoint_dir = os.path.join(path_cfg["checkpoints_dir"], run_name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    # Launch Training
    train(model, dataloader, optimizer, device, config, run_checkpoint_dir, run_name)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()