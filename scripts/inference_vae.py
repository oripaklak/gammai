import os
import sys
import datetime
import torch
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helpers import load_config, setup_device, plot_and_save_spectra
from models.vae import SpectrumVAE

# --- 1. Load Configuration ---
config = load_config("configs/vae.yaml")

LATENT_DIM = config["model"]["latent_dim"]
WEIGHTS_PATH = config["generation"]["model_path"]
NUM_SAMPLES = config["generation"]["num_samples_to_generate"]
SCALING_PATH = config["paths"]["scaling_factor_path"]
L = config["model"]["signal_len"]

device = setup_device(config["hardware"])

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
generated_counts = unnormalize(generated_scaled, max_val)
generated_counts = np.clip(np.round(generated_counts), a_min=0, a_max=None)

# --- 5. Setup Directory ---
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = os.path.splitext(os.path.basename(WEIGHTS_PATH))[0]
run_name = os.path.basename(os.path.dirname(WEIGHTS_PATH))
save_dir = f"results/{run_name}/{model_name}_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

# --- 6. Plot and Save ---
plot_and_save_spectra(generated_counts, save_dir, model_name)

# Save the raw numpy array
data_path = os.path.join(save_dir, f"all_{NUM_SAMPLES}_samples.npy")
np.save(data_path, generated_counts)
print(f"Successfully saved all {NUM_SAMPLES} images and data to: {save_dir}/")