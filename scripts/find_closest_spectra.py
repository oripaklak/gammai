import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from argparse import ArgumentParser
from pathlib import Path


def load_original_txt_data(data_dirs, target_length):
    """
    Loads all txt files from the specified directories,
    forces them to target_length, and returns a numpy matrix.
    """
    real_data_list = []
    file_paths = []

    print("Reading original .txt files and standardizing lengths... (this takes a few seconds)")
    for d in data_dirs:
        folder_path = Path(d)
        if not folder_path.exists():
            print(f"Warning: Folder {d} not found.")
            continue

        for file_path in sorted(folder_path.glob("*.txt")):
            try:
                # Load the data, skipping the "Channel" header
                spectrum = np.loadtxt(file_path, dtype=np.float32, skiprows=1, usecols=1)

                # --- NEW: Force to exactly target_length ---
                current_len = len(spectrum)
                if current_len < target_length:
                    # Pad with zeros at the end if it's too short
                    spectrum = np.pad(spectrum, (0, target_length - current_len), 'constant')
                elif current_len > target_length:
                    # Truncate if it's too long
                    spectrum = spectrum[:target_length]
                # -------------------------------------------

                real_data_list.append(spectrum)
                file_paths.append(str(file_path))
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

    return np.array(real_data_list), file_paths

def find_and_plot_closest_spectra(results_dir):
    """
    Loads generated spectra, compares them to the original txt dataset using Euclidean distance,
    and plots them side-by-side with their nearest real neighbor and its exact path.
    """
    # 1. Locate the generated data
    gen_files = [f for f in os.listdir(results_dir) if f.startswith("all_") and f.endswith(".npy")]
    if not gen_files:
        raise FileNotFoundError(f"No generated .npy file found in {results_dir}")
    gen_data_path = os.path.join(results_dir, gen_files[0])

    print(f"Loading generated data: {gen_data_path}")
    gen_data = np.load(gen_data_path).astype(np.float32)
    num_samples = gen_data.shape[0]

    # 2. Load the original dataset straight from the TXT directories
    data_dirs = [
        "data/IAEA_HPGe/2024/txt",
        "data/IAEA_HPGe/2023/txt"
    ]

    # We pass the exact length of the generated spectra (e.g., 16384)
    target_len = gen_data.shape[1]
    real_data, file_paths = load_original_txt_data(data_dirs, target_length=target_len)

    if len(real_data) == 0:
        raise ValueError("No .txt files were successfully loaded. Check your paths.")

    print(f"Loaded {len(real_data)} real spectra. Calculating distances...")

    # 3. Calculate pairwise Euclidean distances
    distances = cdist(gen_data, real_data, metric='euclidean')

    # 4. Find the index of the minimum distance for each generated sample
    closest_indices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)

    # 5. Create a directory to save the comparison plots
    retrieval_dir = os.path.join(results_dir, "retrieval_comparisons")
    os.makedirs(retrieval_dir, exist_ok=True)

    print(f"Plotting and saving {num_samples} comparisons to {retrieval_dir}/")

    # 6. Save a text log of the matches with the exact file paths!
    with open(os.path.join(retrieval_dir, "retrieval_log.txt"), "w") as f:
        f.write("Generated_Index | Euclidean_Distance | Closest_Real_File\n")
        f.write("-" * 80 + "\n")

        for i in range(num_samples):
            gen_spectrum = gen_data[i]
            real_idx = closest_indices[i]
            real_spectrum = real_data[real_idx]
            dist = min_distances[i]
            closest_file_path = file_paths[real_idx]

            # Write to log
            f.write(f"{i:03d} | {dist:18.2f} | {closest_file_path}\n")

            # Plot
            _plot_side_by_side(gen_spectrum, real_spectrum, closest_file_path, dist, i, retrieval_dir)

    print("Retrieval analysis complete!")


def _plot_side_by_side(gen_spec, real_spec, closest_file_path, dist, gen_idx, save_dir):
    """Helper function to plot the generated vs real spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Extract just the filename (e.g., 'spectrum_01.txt') from the full path for a cleaner title
    file_name = os.path.basename(closest_file_path)

    # --- Plot Generated ---
    axes[0].plot(gen_spec, color='blue', linewidth=1)
    axes[0].set_yscale('log')
    axes[0].set_title(f"Generated Sample {gen_idx:03d}")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Counts (Log Scale)")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)
    _crop_x_axis(axes[0], gen_spec)

    # --- Plot Real ---
    axes[1].plot(real_spec, color='green', linewidth=1)
    axes[1].set_yscale('log')
    # Added the specific file name and distance to the title
    axes[1].set_title(f"Closest Real: {file_name}\nDistance: {dist:.2f}")
    axes[1].set_xlabel("Channel")
    # Added a text box at the bottom to show the FULL path
    axes[1].text(0.5, -0.15, f"Path: {closest_file_path}",
                 horizontalalignment='center', verticalalignment='center',
                 transform=axes[1].transAxes, fontsize=9, color='gray')
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    _crop_x_axis(axes[1], real_spec)

    # Adjust layout so the path text isn't cut off
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(save_dir, f"comparison_{gen_idx:03d}.png"), dpi=300)
    plt.close(fig)


def _crop_x_axis(ax, spec):
    """Crops the X-axis slightly past the last non-zero value for cleaner plots."""
    non_zero_indices = np.nonzero(spec)[0]
    if len(non_zero_indices) > 0:
        max_x = non_zero_indices[-1]
        ax.set_xlim(0, max_x + 50)
    else:
        ax.set_xlim(0, len(spec))


if __name__ == "__main__":
    parser = ArgumentParser(description='Find closest real spectra for generated samples directly from txt files')
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to the specific inference results folder (e.g., results/diff_...)")

    args = parser.parse_args()
    find_and_plot_closest_spectra(args.results_dir)