import os
import datetime
import argparse
import torch
import torch.optim as optim
import numpy as np
import wandb
from torch.utils.data import TensorDataset, DataLoader

from utils import load_config, setup_device, set_seed
from models.vae import SpectrumVAE, vae_loss_function


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Spectrum VAE")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    args = parser.parse_args()

    # Load configuration blocks
    config = load_config(args.config)
    train_cfg = config["training"]
    model_cfg = config["model"]
    path_cfg = config["paths"]
    track_cfg = config["tracking"]["wandb"]

    # 1. Set Random Seeds for Reproducibility
    set_seed(train_cfg["seed"])

    # 2. Build Descriptive Naming String
    run_name = f"vae_ld{model_cfg['latent_dim']}_bs{train_cfg['batch_size']}_lr{train_cfg['learning_rate']}_ep{train_cfg['epochs']}"

    # 3. Initialize Weights & Biases if enabled
    if track_cfg.get("enabled", True):
        wandb.init(
            project=config["experiment"]["project_name"],
            name=run_name,
            config=config  # Logs the entire config dictionary
        )

    # 4. Load and Normalize the Data
    print(f"Loading data from {path_cfg['data_path']}...")
    matrix = np.load(path_cfg["data_path"])
    N, L = matrix.shape

    log_matrix = np.log1p(matrix)
    max_val = np.max(log_matrix)
    scaled_matrix = log_matrix / max_val

    # Save scaling factor for generation/inference step
    os.makedirs(os.path.dirname(path_cfg["scaling_factor_path"]), exist_ok=True)
    np.save(path_cfg["scaling_factor_path"], max_val)

    # 5. Prepare PyTorch DataLoaders
    tensor_x = torch.Tensor(scaled_matrix)
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True)

    # 6. Initialize Device and Model
    device = setup_device(config["hardware"])
    model = SpectrumVAE(L=L, latent_dim=model_cfg["latent_dim"]).to(device)

    # Handle multi-GPU wrapping if requested
    if config["hardware"].get("num_gpus", 1) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    # Watch the model gradients/topology via W&B
    if track_cfg.get("enabled", True) and track_cfg.get("log_model_graph", True):
        wandb.watch(model, log="all", log_freq=track_cfg.get("watch_log_freq", 10))

    # 7. Training Loop
    print(f"Starting training for {train_cfg['epochs']} epochs...")
    for epoch in range(train_cfg["epochs"]):
        model.train()
        train_loss = 0

        for batch in dataloader:
            batch_data = batch[0].to(device)

            optimizer.zero_grad()
            reconstructed_batch, mu, logvar = model(batch_data)
            loss = vae_loss_function(reconstructed_batch, batch_data, mu, logvar)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_train_loss = train_loss / len(dataloader) # divide by number of batches
        current_epoch = epoch + 1

        # Base metrics for W&B
        metrics = {
            "epoch": current_epoch,
            "train_loss": avg_train_loss
        }

        # --- Evaluation Placeholder ---
        #if current_epoch % train_cfg.get("eval_freq", 20) == 0:
            # Uncomment and implement your validation logic here:
            # print(f"--- Running Evaluation at Epoch {current_epoch} ---")
            # val_loss = run_eval_pipeline(model, val_loader, device)
            # metrics["val_loss"] = val_loss
            #pass

        # Log metrics to W&B
        if track_cfg.get("enabled", True):
            wandb.log(metrics)

        # Print to console according to print_freq
        if current_epoch % train_cfg.get("print_freq", 10) == 0:
            print(f"Epoch [{current_epoch}/{train_cfg['epochs']}] | Train Loss: {avg_train_loss:.4f}")

    # 8. Save the trained model weights
    os.makedirs(path_cfg["checkpoints_dir"], exist_ok=True)
    save_path = os.path.join(path_cfg["checkpoints_dir"], f"{run_name}.pth")

    # Unwrap model state_dict safely if DataParallel was used
    state_dict_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state_dict_to_save, save_path)

    print(f"Training complete. Model saved locally to {save_path}")

    # Finish W&B run
    if track_cfg.get("enabled", True):
        wandb.finish()


if __name__ == "__main__":
    main()