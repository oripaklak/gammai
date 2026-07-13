import os
import datetime
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from models.vae import SpectrumVAE

# --- 1. Load Configuration ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

LATENT_DIM = config["model"]["latent_dim"]
WEIGHTS_PATH = config["generation"]["weights_to_use"]
NUM_SAMPLES = config["generation"]["num_samples_to_generate"]
SCALING_PATH = config["paths"]["scaling_factor_path"]
GPU_ID = config["hardware"]["device_id"]
L = 16384

if torch.cuda.is_available():
    device = torch.device(f"cuda:{GPU_ID}")
else:
    device = torch.device("cpu")

# --- 2. Load the Scaling Factor and Model Weights ---
max_val = np.load(SCALING_PATH)

model = SpectrumVAE(L=L, latent_dim=LATENT_DIM).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

# --- 3. Generate New Spectra ---
print(f"Sampling {NUM_SAMPLES} spectra from latent space...")
with torch.no_grad():
    z = torch.randn(NUM_SAMPLES, LATENT_DIM).to(device)
    generated_scaled = model.decode(z).cpu().numpy()

    if generated_scaled.ndim == 3:
        generated_scaled = generated_scaled.squeeze(axis=1)
    elif generated_scaled.ndim == 1:
        generated_scaled = np.expand_dims(generated_scaled, axis=0)

# --- 4. Reverse the Normalization ---
generated_log = generated_scaled * max_val
generated_counts = np.expm1(generated_log)
generated_counts = np.clip(np.round(generated_counts), a_min=0, a_max=None)

# --- 5. Setup Directory ---
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Extract the checkpoint name (e.g., 'checkpoint_ep43000')
model_name = os.path.splitext(os.path.basename(WEIGHTS_PATH))[0]

# Extract the parent directory name (e.g., 'vae_ld32_bs16_lr0.001_ep100000')
run_name = os.path.basename(os.path.dirname(WEIGHTS_PATH))

# Construct the new save path
save_dir = f"results/{run_name}/{model_name}_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

# --- 6. Plot and Save ALL Images Silently ---
print(f"Saving {NUM_SAMPLES} plots to {save_dir}/ ...")

for i in range(NUM_SAMPLES):
    counts = generated_counts[i]

    plt.figure(figsize=(10, 5))
    plt.plot(counts, color='blue', linewidth=1)

    # --- Find the last non-zero channel ---
    non_zero_indices = np.nonzero(counts)[0]
    if len(non_zero_indices) > 0:
        max_x = non_zero_indices[-1]
        # Crop the X-axis up to the last value + 50 channels for aesthetic padding
        plt.xlim(0, max_x + 50)
    else:
        # Edge case fallback if the entire array is zeros (very rare)
        plt.xlim(0, L)
        # -----------------------------------

    plt.yscale('log')
    plt.title(f"Generated Spectrum {i + 1} of {NUM_SAMPLES}\nModel: {model_name}")
    plt.xlabel("Channel")
    plt.ylabel("Counts (Log Scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    # Save each plot
    plot_path = os.path.join(save_dir, f"spectrum_{i + 1:03d}.png")
    plt.savefig(plot_path, dpi=300)

    plt.close()

# Save the raw numpy array
data_path = os.path.join(save_dir, f"all_{NUM_SAMPLES}_samples.npy")
np.save(data_path, generated_counts)

print(f"Successfully saved all {NUM_SAMPLES} images and data to: {save_dir}/")