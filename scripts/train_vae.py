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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Spectrum VAE")
    parser.add_argument("--exp", type=str, default="configs/experiments/vae_base.yaml", help="Path to the experiment config")
    parser.add_argument("--model", type=str, default="configs/models/vae.yaml", help="Path to the model config")
    args = parser.parse_args()

    # Load and merge configuration blocks
    config = load_config(args.exp, args.model)
    train_cfg = config["training"]
    model_cfg = config["model"]
    path_cfg = config["paths"]
    track_cfg = config["tracking"]["wandb"]
    loss_cfg = config["loss"]

    # 1. Set Random Seeds for Reproducibility
    set_seed(train_cfg["seed"])

    # 2. Build Descriptive Naming String
    run_name = f"vae_ld{model_cfg['latent_dim']}_bs{train_cfg['batch_size']}_lr{train_cfg['learning_rate']}_ep{train_cfg['epochs']}"

    # 3. Initialize Weights & Biases if enabled
    if track_cfg.get("enabled", True):
        wandb.init(
            project=config["experiment"]["project_name"],
            name=run_name,
            config=config,  # Logs the entire config dictionary
        )

        # 4 & 5. Load and Normalize Data using centralized dataset
        print(f"Loading data from {path_cfg['data_path']}...")
        dataloader = from_path([path_cfg["data_path"]], config)

        # Extract dimensions and scaling factor from the dataset instance
        dataset_instance = dataloader.dataset
        N, L = dataset_instance.matrix.shape

        # Save scaling factor for generation/inference step
        os.makedirs(os.path.dirname(path_cfg["scaling_factor_path"]), exist_ok=True)
        np.save(path_cfg["scaling_factor_path"], dataset_instance.max_val)

    # 6. Initialize Device and Model
    device = setup_device(config["hardware"])
    model = SpectrumVAE(L=L, latent_dim=model_cfg["latent_dim"]).to(device)

    # number of total parameters and log it
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")

    # Handle multi-GPU wrapping if requested
    if config["hardware"].get("num_gpus", 1) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    # Watch the model gradients/topology via W&B
    if track_cfg.get("enabled", True) and track_cfg.get("log_model_graph", True):
        wandb.watch(model, log="all", log_freq=track_cfg.get("watch_log_freq", 10))

    # --- SETUP CHECKPOINT DIRECTORY ---
    # Create a dedicated subdirectory for this specific run inside your checkpoints folder
    run_checkpoint_dir = os.path.join(path_cfg["checkpoints_dir"], run_name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    # 7. Training Loop
    print(f"Starting training for {train_cfg['epochs']} epochs...")
    global_step = 0  # Add a counter to track individual steps across all epochs

    for epoch in range(train_cfg["epochs"]):
        model.train()
        train_loss = 0

        for batch in dataloader:
            batch_data = batch['signal'].to(device)

            optimizer.zero_grad()
            reconstructed_batch, mu, logvar = model(batch_data)
            loss = vae_loss_function(reconstructed_batch, batch_data, mu, logvar, loss_cfg["beta"])

            loss.backward()

            # Add Gradient Norm Calculation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)

            step_loss = loss.item()
            train_loss += step_loss
            optimizer.step()

            global_step += 1

        avg_train_loss = train_loss / len(dataloader.dataset)
        current_epoch = epoch + 1

        # --- Standardized Epoch Metrics ---
        metrics = {
            "train/epoch": current_epoch,
            "train/loss_vae": avg_train_loss,
            "train/global_step": global_step,
            "train/grad_norm_vae": grad_norm
        }

        # --- Evaluation Pipeline ---
        if current_epoch % train_cfg.get("eval_freq", 20) == 0:
            model.eval()
            all_real_data = []
            all_fake_data = []

            with torch.no_grad():
                # We need a decent sample size to compute reliable statistics
                # We'll run through the dataloader to grab real data and generate matching fake data
                for val_batch in dataloader:
                    real_batch = val_batch['signal'].to(device)

                    # Purely generated data from random latent space
                    z = torch.randn(real_batch.size(0), model_cfg["latent_dim"]).to(device)
                    generated_batch = model.decode(z).squeeze(1) # ADDED

                    all_real_data.append(real_batch.cpu().numpy())
                    all_fake_data.append(generated_batch.cpu().numpy())

            # Concatenate batches into full datasets
            real_np = np.concatenate(all_real_data, axis=0)
            fake_np = np.concatenate(all_fake_data, axis=0)

            # Compute Metrics
            try:
                fd_score = calculate_frechet_distance(real_np, fake_np)
                kid_score = calculate_polynomial_mmd(real_np, fake_np)

                metrics["eval/FID"] = fd_score
                metrics["eval/KID"] = kid_score

                # --- Add Image Logging ---
                fig, ax = plt.subplots()
                ax.plot(fake_np[0])  # Plot the first generated spectrum
                ax.set_title(f'VAE Generated Signal (Epoch {current_epoch})')
                metrics["feature/signal_vae"] = wandb.Image(fig)
                plt.close(fig)
                # -------------------------

                print(f"--> Eval Epoch {current_epoch} | FD: {fd_score:.4f} | KD: {kid_score:.4f}")
            except Exception as e:
                print(f"Warning: Failed to calculate evaluation metrics. Error: {e}")

            model.train()  # Set back to train mode!

        if track_cfg.get("enabled", True):
            wandb.log(metrics)

        # Print to console according to print_freq
        if current_epoch % train_cfg.get("print_freq", 10) == 0:
            print(f"Epoch [{current_epoch}/{train_cfg['epochs']}] | Train Loss: {avg_train_loss:.4f}")

        # --- SAVE INTERMEDIATE CHECKPOINTS ---
        # Default to 1000 if not specified in config
        ckpt_freq = train_cfg.get("checkpoint_freq", 1000)

        if current_epoch % ckpt_freq == 0:
            ckpt_name = f"checkpoint_ep{current_epoch}.pth"
            ckpt_save_path = os.path.join(run_checkpoint_dir, ckpt_name)

            # Unwrap model state_dict safely if DataParallel was used
            state_dict_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict_to_save, ckpt_save_path)

            print(f"--- Saved intermediate checkpoint: {ckpt_save_path} ---")

    # 8. Save the final trained model weights at the end of training
    final_save_path = os.path.join(run_checkpoint_dir, f"{run_name}_final.pth")
    state_dict_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state_dict_to_save, final_save_path)

    print(f"Training complete. Final model saved locally to: {final_save_path}")
    print("Waiting 5 seconds for file I/O to finish...")
    time.sleep(5)
    os._exit(0)

    # Finish W&B run - firewall creates an error
    #if track_cfg.get("enabled", True):
    #    wandb.finish()


if __name__ == "__main__":
    main()