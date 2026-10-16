import os
import sys
import datetime
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helpers import load_config, setup_device
from utils.dataset import from_path
from models.vae import SpectrumVAE


def plot_spectra_manifold(ax, embeddings, signals, labels, num_to_plot=800, zoom=0.06):
    """
    Overlays miniature line plots of spectra over the 2D scatter plot.
    Colors the background points and the bounding boxes based on cluster labels.
    """
    # Define a colormap for the clusters (tab10 gives 10 distinct categorical colors)
    cmap = plt.get_cmap('tab10')

    # Plot the full background scatter, color-coded by cluster label
    ax.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.3, c=labels, cmap=cmap, s=10)

    # Randomly sample points to overlay the actual spectrum lines
    indices = np.random.choice(len(embeddings), size=min(num_to_plot, len(embeddings)), replace=False)

    x_range = embeddings[:, 0].max() - embeddings[:, 0].min()
    y_range = embeddings[:, 1].max() - embeddings[:, 1].min()

    w = x_range * zoom  # Width of miniature plot box
    h = y_range * zoom  # Height of miniature plot box

    for idx in indices:
        x, y = embeddings[idx]
        counts = signals[idx]
        cluster_id = labels[idx]

        # Get the assigned color for this specific cluster
        cluster_color = cmap(cluster_id % 10)

        # --- Create a miniature inset plot at the (x, y) data coordinates ---
        axins = ax.inset_axes([x - w / 2, y - h / 2, w, h], transform=ax.transData)
        axins.set_facecolor('white')

        # --- Apply the plotting logic ---
        axins.plot(counts, color='blue', linewidth=0.5)
        axins.set_yscale('log')

        # Find the last non-zero channel to crop the X-axis
        non_zero_indices = np.nonzero(counts)[0]
        if len(non_zero_indices) > 0:
            max_x = non_zero_indices[-1]
            axins.set_xlim(0, max_x + 50)
        else:
            axins.set_xlim(0, len(counts))

        # Add the grid
        axins.grid(True, which="both", ls="--", alpha=0.5, linewidth=0.2)

        # Hide tick labels so it doesn't get cluttered
        axins.tick_params(axis='both', which='both', bottom=False, top=False,
                          left=False, right=False, labelbottom=False, labelleft=False)

        # --- Color the bounding box based on the cluster ---
        for spine in axins.spines.values():
            spine.set_color(cluster_color)
            spine.set_linewidth(1.5)  # Make the border thicker so the color pops
            spine.set_zorder(10)


def main():
    # --- 1. Load Configurations via Argparse ---
    parser = argparse.ArgumentParser(description="Plot VAE Latent Space")
    parser.add_argument("--exp", type=str, default="configs/experiments/vae_base.yaml",
                        help="Path to the experiment config")
    parser.add_argument("--model", type=str, default="configs/models/vae.yaml",
                        help="Path to the model config")
    parser.add_argument("--clusters", type=int, default=5,
                        help="Number of clusters to group the latent space into")
    args = parser.parse_args()

    config = load_config(args.exp, args.model)

    # Extract config variables
    LATENT_DIM = config["model"]["latent_dim"]
    L = config["model"]["signal_len"]
    WEIGHTS_PATH = config["generation"]["model_path"]
    SCALING_PATH = config["paths"]["scaling_factor_path"]
    DATA_PATH = config["paths"]["data_path"]

    device = setup_device(config["hardware"])

    # --- 2. Load Model and Data ---
    model = SpectrumVAE(L=L, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    print(f"Loading data from {DATA_PATH}...")
    dataloader = from_path([DATA_PATH], config)

    max_val = np.load(SCALING_PATH)

    # --- 3. Encode Data to Latent Space ---
    print("Encoding dataset into latent space...")
    all_mu = []
    all_signals = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['signal'].to(device)
            x_unsqueezed = x.unsqueeze(1)

            # Get the latent mean (mu)
            mu, _ = model.encode(x_unsqueezed)

            all_mu.append(mu.cpu().numpy())
            all_signals.append(x.cpu().numpy())

    all_mu = np.concatenate(all_mu, axis=0)
    all_signals = np.concatenate(all_signals, axis=0)

    # --- 4. Dimensionality Reduction (if needed) ---
    if LATENT_DIM > 2:
        print(f"Latent dim is {LATENT_DIM}. Applying PCA to reduce to 2D...")
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_mu)
    else:
        print("Latent dim is already 2. No dimensionality reduction needed.")
        embeddings_2d = all_mu

    # --- 5. Clustering ---
    print(f"Applying K-Means clustering with k={args.clusters}...")
    kmeans = KMeans(n_clusters=args.clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings_2d)

    # --- 6. Reverse Normalization for Plotted Signals ---
    signals_log = all_signals * max_val
    signals_counts = np.expm1(signals_log)
    signals_counts = np.clip(np.round(signals_counts), a_min=0, a_max=None)

    # --- 7. Plotting ---
    print("Generating Latent Space cluster plot...")
    fig, ax = plt.subplots(figsize=(14, 10))

    plot_spectra_manifold(ax, embeddings_2d, signals_counts, cluster_labels, num_to_plot=800, zoom=0.06)

    ax.set_title(f"VAE Latent Space Clusters (L={L}, Latent Dim={LATENT_DIM}, Clusters={args.clusters})")
    ax.set_xlabel("Latent Dimension 1 (PCA)" if LATENT_DIM > 2 else "Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2 (PCA)" if LATENT_DIM > 2 else "Latent Dimension 2")

    # --- 8. Save Results ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = os.path.splitext(os.path.basename(WEIGHTS_PATH))[0]
    run_name = os.path.basename(os.path.dirname(WEIGHTS_PATH))

    save_dir = f"results/{run_name}/{model_name}_{timestamp}_latent_viz"
    os.makedirs(save_dir, exist_ok=True)

    plot_path = os.path.join(save_dir, "latent_space_clusters.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Successfully saved latent space visualization to: {plot_path}")


if __name__ == "__main__":
    main()