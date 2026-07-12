import os
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.vae import SpectrumVAE  # Updated to look inside the models/ folder

# --- Configuration ---
L = 16384  # Make sure this matches your matrix's max channel length!
LATENT_DIM = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the Scaling Factor and Model Weights
max_val = np.load("data/scaling_factor.npy")

model = SpectrumVAE(L=L, latent_dim=LATENT_DIM).to(device)
# Updated to look inside the checkpoints/ directory for the weights
model.load_state_dict(torch.load("checkpoints/vae_weights.pth", map_location=device))
model.eval()

# 2. Generate a New Spectrum
print("Sampling from latent space...")
with torch.no_grad():
    # Sample a 1 x LATENT_DIM matrix from a standard normal distribution (mean=0, std=1)
    z = torch.randn(1, LATENT_DIM).to(device)

    # Decode the latent vector into a spectrum (scaled between 0 and 1)
    generated_scaled = model.decode(z).cpu().numpy().squeeze()

# 3. Reverse the Normalization
# First, multiply by max_val to get back to the log scale
generated_log = generated_scaled * max_val
# Second, use exp(x) - 1 (expm1) to reverse the log1p, getting real integer counts
generated_counts = np.expm1(generated_log)
# Ensure no negative counts and round to integers
generated_counts = np.clip(np.round(generated_counts), a_min=0, a_max=None)

# --- 4. Setup Directory and Timestamp ---
# Ensure the 'results' directory exists
os.makedirs("results", exist_ok=True)

# Generate a unique timestamp (Year-Month-Day_Hour-Minute-Second)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_prefix = f"results/generated_spectrum_{timestamp}"

# --- 5. Plot and Save the Generated Spectrum ---
plt.figure(figsize=(10, 5))
plt.plot(generated_counts, color='blue', linewidth=1)

# Formatting the plot to look like standard spectroscopy software
plt.yscale('log')  # Gamma spectra are almost always viewed in log scale!
plt.title("AI-Generated HPGe Gamma Spectrum")
plt.xlabel("Channel")
plt.ylabel("Counts (Log Scale)")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()

# Save the plot as a PNG image
plot_path = f"{file_prefix}.png"
plt.savefig(plot_path, dpi=300)

# Save the raw numpy array (highly recommended for future analysis)
data_path = f"{file_prefix}.npy"
np.save(data_path, generated_counts)

print(f"Successfully saved plot to: {plot_path}")
print(f"Successfully saved data to: {data_path}")

# Show the plot
plt.show()