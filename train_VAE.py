import os
import datetime
import torch
import torch.optim as optim
import numpy as np
import wandb
from torch.utils.data import TensorDataset, DataLoader
from models.vae import SpectrumVAE, vae_loss_function

# --- Hyperparameters ---
CONFIG = {
    "batch_size": 16,
    "epochs": 300,
    "learning_rate": 1e-3,
    "latent_dim": 32
}

# Generate a unique timestamp for injectivity (Year-Month-Day_Hour-Minute-Second)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
unique_run_name = f"baseline-vae-run_{timestamp}"

# 1. Initialize Weights & Biases
wandb.init(
    project="hpge-spectrum-vae",
    name=unique_run_name,  # Updated to use the unique timestamped name
    config=CONFIG
)

# 2. Load and Normalize the Data
print("Loading data...")
matrix = np.load("data/spectrum_matrix.npy")
N, L = matrix.shape

# Apply log1p scaling and Min-Max scaling
log_matrix = np.log1p(matrix) # log(1+x)
max_val = np.max(log_matrix)
scaled_matrix = log_matrix / max_val
np.save("data/scaling_factor.npy", max_val)

# 3. Prepare PyTorch DataLoaders
tensor_x = torch.Tensor(scaled_matrix)
dataset = TensorDataset(tensor_x)
dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

# 4. Initialize the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}...")

# Updated model initialization (removed hidden_dim)
model = SpectrumVAE(L=L, latent_dim=CONFIG["latent_dim"]).to(device)
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

# Tell W&B to watch the model's gradients and weights
wandb.watch(model, log="all", log_freq=10)

# 5. Training Loop
model.train()
for epoch in range(CONFIG["epochs"]):
    train_loss = 0
    for batch in dataloader:
        batch_data = batch[0].to(device)

        optimizer.zero_grad()

        # Forward pass (the Conv1D reshaping happens automatically inside)
        reconstructed_batch, mu, logvar = model(batch_data)

        # Calculate loss
        loss = vae_loss_function(reconstructed_batch, batch_data, mu, logvar)

        # Backward pass and optimize
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_loss = train_loss / len(dataloader.dataset)

    # Log metrics to W&B
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss
    })

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{CONFIG['epochs']}] | Average Loss: {avg_loss:.4f}")

# 6. Save the trained model and finish W&B run
# Ensure checkpoints directory exists
os.makedirs("checkpoints", exist_ok=True)

# Updated save path to prevent overwriting previous runs
save_path = f"checkpoints/vae_weights_{timestamp}.pth"

torch.save(model.state_dict(), save_path)

# Log the final model file to W&B as an artifact
artifact = wandb.Artifact('vae_model', type='model')
artifact.add_file(save_path)
wandb.log_artifact(artifact)

print(f"Training complete. Model saved locally to {save_path} and to W&B.")
wandb.finish()