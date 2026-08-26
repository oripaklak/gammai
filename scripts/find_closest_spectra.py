import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from argparse import ArgumentParser
from pathlib import Path


def load_grouped_real_data(data_dirs):
    """
    Loads all txt files and groups them by their standard detector
    calibrations (4096, 8192, 16384).
    """
    bins = [4096, 8192, 16384]
    grouped_data = {b: [] for b in bins}
    grouped_paths = {b: [] for b in bins}

    print("Reading original .txt files and grouping by calibration length...")
    for d in data_dirs:
        folder_path = Path(d)
        if not folder_path.exists():
            continue

        for file_path in sorted(folder_path.glob("*.txt")):
            try:
                # Load the data, skipping the "Channel" header
                spectrum = np.loadtxt(file_path, dtype=np.float32, skiprows=1, usecols=1)
                L = len(spectrum)

                # Find which standard bin this file belongs to
                closest_bin = min(bins, key=lambda x: abs(x - L))

                # Clean up minor length discrepancies (e.g. if length is 4095)
                if L < closest_bin:
                    spectrum = np.pad(spectrum, (0, closest_bin - L), 'constant')
                elif L > closest_bin:
                    spectrum = spectrum[:closest_bin]

                grouped_data[closest_bin].append(spectrum)
                grouped_paths[closest_bin].append(str(file_path))
            except Exception as e:
                pass

    # Convert lists to NumPy matrices for fast distance calculation
    for b in bins:
        if len(grouped_data[b]) > 0:
            grouped_data[b] = np.array(grouped_data[b])
            print(f"  -> Found {len(grouped_data[b])} real files for {b} channels")

    return grouped_data, grouped_paths


def find_and_plot_closest_spectra(results_dir):
    gen_files = [f for f in os.listdir(results_dir) if f.startswith("all_") and f.endswith(".npy")]
    if not gen_files:
        raise FileNotFoundError(f"No generated .npy file found in {results_dir}")
    gen_data_path = os.path.join(results_dir, gen_files[0])

    print(f"Loading generated data: {gen_data_path}")
    gen_data = np.load(gen_data_path).astype(np.float32)
    num_samples = gen_data.shape[0]

    data_dirs = [
        "data/IAEA_HPGe/2024/txt",
        "data/IAEA_HPGe/2023/txt",
        "data/Baq_HPGe/txt"
    ]

    # 1. Load and group the real data
    real_data_groups, real_paths_groups = load_grouped_real_data(data_dirs)

    retrieval_dir = os.path.join(results_dir, "retrieval_comparisons")
    os.makedirs(retrieval_dir, exist_ok=True)

    print(f"Plotting and saving {num_samples} comparisons to {retrieval_dir}/")

    with open(os.path.join(retrieval_dir, "retrieval_log.txt"), "w") as f:
        f.write("Gen_Index | Target_Bin | Euclidean_Dist | Closest_Real_File\n")
        f.write("-" * 80 + "\n")

        # 2. Iterate through each generated sample individually
        for i in range(num_samples):
            gen_spec = gen_data[i]

            # Find effective length (last index that is greater than 0)
            non_zeros = np.nonzero(gen_spec)[0]
            eff_len = non_zeros[-1] + 1 if len(non_zeros) > 0 else 0

            # Assign to the closest subgroup (4096, 8192, 16384)
            bins = [4096, 8192, 16384]
            target_bin = min(bins, key=lambda x: abs(x - eff_len))

            # If a specific bin has no real data, fallback to 16384
            if len(real_data_groups[target_bin]) == 0:
                target_bin = 16384

            # Truncate the generated spectrum to match the subgroup bin
            gen_spec_trimmed = gen_spec[:target_bin]

            # 3. Calculate distance ONLY against that specific subgroup
            real_matrix = real_data_groups[target_bin]
            real_paths = real_paths_groups[target_bin]

            # cdist expects 2D arrays, so we wrap gen_spec_trimmed in brackets
            distances = cdist([gen_spec_trimmed], real_matrix, metric='euclidean')[0]

            # Find the winner
            best_idx = np.argmin(distances)
            best_dist = distances[best_idx]
            closest_file_path = real_paths[best_idx]
            best_real_spec = real_matrix[best_idx]

            # Write to log and plot
            f.write(f"{i:03d} | {target_bin:10d} | {best_dist:11.4f} | {closest_file_path}\n")

            # Pass the trimmed version for plotting so the axes match perfectly
            _plot_side_by_side(gen_spec_trimmed, best_real_spec, closest_file_path, best_dist, i+1, retrieval_dir)

    print("Retrieval analysis complete!")


def _plot_side_by_side(gen_spec, real_spec, closest_file_path, dist, gen_idx, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    file_name = os.path.basename(closest_file_path)

    # --- Dynamic Y-Limits Calculation ---
    # Combine both spectra to find a shared min/max range so they scale identically
    combined = np.concatenate([gen_spec, real_spec])
    positive_vals = combined[combined > 0]  # Filter out zeros for log scale

    if len(positive_vals) > 0:
        ymin = max(0.1, float(np.min(positive_vals)) * 0.5)  # Give a little breathing room at the bottom
        ymax = float(np.max(positive_vals)) * 1.5  # Give a little breathing room at the top
    else:
        ymin, ymax = 0.1, 1000.0

    # --- Generated ---
    gen_spec = np.where(gen_spec == 0, 0.1, gen_spec)
    axes[0].plot(gen_spec, color='blue', linewidth=1)
    axes[0].set_yscale('log')
    axes[0].set_title(f"Generated Sample {gen_idx:03d}")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Counts (Log Scale)")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)
    axes[0].set_xlim(0, len(gen_spec))
    axes[0].set_ylim(bottom=ymin, top=ymax)

    # --- Real ---
    real_spec = np.where(real_spec == 0, 0.1, real_spec)
    axes[1].plot(real_spec, color='green', linewidth=1)
    axes[1].set_yscale('log')
    axes[1].set_title(f"Closest Real: {file_name}\nEuclidean  Distance: {dist:.4f}")
    axes[1].set_xlabel("Channel")
    axes[1].text(0.5, -0.15, f"Path: {closest_file_path}",
                 horizontalalignment='center', verticalalignment='center',
                 transform=axes[1].transAxes, fontsize=9, color='gray')
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].set_xlim(0, len(real_spec))
    axes[1].set_ylim(bottom=ymin, top=ymax)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(save_dir, f"comparison_{gen_idx:03d}.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description='Find closest real spectra directly from txt files using subgrouping')
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to the specific inference results folder")

    args = parser.parse_args()
    find_and_plot_closest_spectra(args.results_dir)