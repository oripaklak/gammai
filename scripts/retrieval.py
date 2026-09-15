import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from argparse import ArgumentParser
from pathlib import Path

# Add project root to path so we can import config loaders and models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import load_config
from utils.sift import TemporalSIFT, SIFTBagOfWords


def compute_chi_square_distances(query_hist: np.ndarray, dataset_hists: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Vectorized Chi-Square distance between a single SIFT query histogram (K,)
    and a dataset of histograms (N, K).
    """
    diff = query_hist - dataset_hists
    denom = query_hist + dataset_hists + eps
    return 0.5 * np.sum((diff ** 2) / denom, axis=1)


def load_grouped_real_data(data_dirs, is_calib=False):
    bins = [4096, 8192, 16384]
    grouped_data = {b: [] for b in bins}
    grouped_coeffs = {b: [] for b in bins} if is_calib else None
    grouped_paths = {b: [] for b in bins}

    print(f"Reading original .txt files (Calibration Mode: {is_calib})...")
    for d in data_dirs:
        folder_path = Path(d)
        if not folder_path.exists():
            continue

        coeff_dir = folder_path.parent / "coeff_txt" if is_calib else None

        for file_path in sorted(folder_path.glob("*.txt")):
            if is_calib:
                coeff_path = coeff_dir / file_path.name
                if not coeff_path.exists():
                    continue

            try:
                # 1. Load Spectrum
                spectrum = np.loadtxt(file_path, dtype=np.float32, skiprows=1, usecols=1)
                L = len(spectrum)
                closest_bin = min(bins, key=lambda x: abs(x - L))

                if L < closest_bin:
                    spectrum = np.pad(spectrum, (0, closest_bin - L), 'constant')
                elif L > closest_bin:
                    spectrum = spectrum[:closest_bin]

                # 2. Load Coefficients (if enabled)
                if is_calib:
                    coeffs = np.zeros(4, dtype=np.float32)
                    with open(coeff_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.split(":")
                            if len(parts) == 2:
                                key, val = parts[0].strip(), float(parts[1].strip())
                                if key == 'c0':
                                    coeffs[0] = val
                                elif key == 'c1':
                                    coeffs[1] = val
                                elif key == 'c2':
                                    coeffs[2] = val
                                elif key == 'c3':
                                    coeffs[3] = val
                    grouped_coeffs[closest_bin].append(coeffs)

                grouped_data[closest_bin].append(spectrum)
                grouped_paths[closest_bin].append(str(file_path))
            except Exception:
                pass

    for b in bins:
        if len(grouped_data[b]) > 0:
            grouped_data[b] = np.array(grouped_data[b])
            if is_calib:
                grouped_coeffs[b] = np.array(grouped_coeffs[b])
            print(f"  -> Found {len(grouped_data[b])} real files for {b} channels")

    return grouped_data, grouped_coeffs, grouped_paths


def find_and_plot_closest_spectra(data_dirs, results_dir, config, metric="euclidean", calib_weight=0.1, n_clusters=64):
    is_calib = config.get('calibration', {}).get('enabled', False)

    # 1. Load Generated Spectra
    gen_files = [f for f in os.listdir(results_dir) if f.startswith("all_") and f.endswith("samples.npy")]
    if not gen_files:
        raise FileNotFoundError(f"No generated samples .npy file found in {results_dir}")
    gen_data = np.load(os.path.join(results_dir, gen_files[0])).astype(np.float32)
    num_samples = gen_data.shape[0]

    # 2. Load Generated Coefficients (if enabled)
    gen_coeffs = None
    if is_calib:
        coeff_files = [f for f in os.listdir(results_dir) if f.startswith("all_") and f.endswith("coeffs.npy")]
        if not coeff_files:
            raise FileNotFoundError(f"No generated coeffs .npy file found in {results_dir}")
        gen_coeffs = np.load(os.path.join(results_dir, coeff_files[0])).astype(np.float32)

    # 3. Load Real Dataset
    real_data_groups, real_coeffs_groups, real_paths_groups = load_grouped_real_data(data_dirs, is_calib)

    # 4. SIFT Codebook Initialization (if SIFT mode is selected)
    bow_model = None
    real_sift_groups = {}
    if metric == "sift":
        print("\nInitializing 1D SIFT Extractor & Codebook...")

        # Dynamically retrieve dataset_name from config
        dataset_name = config.get("paths", {}).get("dataset_name", "dataset")

        # Setup output directory under data/codebooks/
        codebook_dir = Path("data/codebooks")
        codebook_dir.mkdir(parents=True, exist_ok=True)
        codebook_path = codebook_dir / f"sift_codebook_{dataset_name}_k{n_clusters}.pkl"

        sift_extractor = TemporalSIFT(tau_step=15, num_scales=6, n_b=4, a=2)
        bow_model = SIFTBagOfWords(extractor=sift_extractor, n_clusters=n_clusters)

        # Check if pre-trained codebook already exists
        if codebook_path.exists():
            print(f"Loading existing SIFT Codebook from '{codebook_path}'...")
            bow_model.load_codebook(str(codebook_path))
        else:
            print(f"Codebook not found at '{codebook_path}'. Training new Codebook on real dataset...")
            all_real_signals = []
            for b in real_data_groups:
                if len(real_data_groups[b]) > 0:
                    all_real_signals.extend(real_data_groups[b])

            bow_model.fit_codebook(all_real_signals)
            bow_model.save_codebook(str(codebook_path))

        print("Pre-encoding real spectra to SIFT histograms...")
        for b in real_data_groups:
            if len(real_data_groups[b]) > 0:
                real_sift_groups[b] = np.array([bow_model.signal_to_histogram(sig) for sig in real_data_groups[b]])
            else:
                real_sift_groups[b] = np.empty((0, n_clusters))

    # 5. Output Directory Setup
    mode_str = "calibrated" if is_calib else "uncalibrated"
    retrieval_dir = os.path.join(results_dir, f"retrieval_comparisons_{metric}_{mode_str}")
    os.makedirs(retrieval_dir, exist_ok=True)

    dist_label = "SIFT_Chi2_Dist" if metric == "sift" else "Euclidean_Dist"

    with open(os.path.join(retrieval_dir, "retrieval_log.txt"), "w") as f:
        if is_calib:
            f.write(f"Gen_Index | Target_Bin | {dist_label:>14} | Calib_Dist | Total_Score | Closest_Real_File\n")
        else:
            f.write(f"Gen_Index | Target_Bin | {dist_label:>14} | Closest_Real_File\n")
        f.write("-" * 100 + "\n")

        for i in range(num_samples):
            gen_spec = gen_data[i]
            gen_coeff = gen_coeffs[i] if is_calib else None

            non_zeros = np.nonzero(gen_spec)[0]
            eff_len = non_zeros[-1] + 1 if len(non_zeros) > 0 else 0
            bins = [4096, 8192, 16384]
            target_bin = min(bins, key=lambda x: abs(x - eff_len))

            if len(real_data_groups[target_bin]) == 0:
                target_bin = 16384

            gen_spec_trimmed = gen_spec[:target_bin]
            real_matrix = real_data_groups[target_bin]
            real_paths = real_paths_groups[target_bin]

            # Shape Distance Computation
            if metric == "sift":
                gen_sift_hist = bow_model.signal_to_histogram(gen_spec_trimmed)
                shape_distances = compute_chi_square_distances(gen_sift_hist, real_sift_groups[target_bin])
            else:
                shape_distances = cdist([gen_spec_trimmed], real_matrix, metric='euclidean')[0]

            # Calibration Distance Computation (if enabled)
            if is_calib:
                real_coeffs_mat = real_coeffs_groups[target_bin]
                coeff_means = np.mean(real_coeffs_mat, axis=0)
                coeff_stds = np.std(real_coeffs_mat, axis=0) + 1e-8
                gen_coeff_norm = (gen_coeff - coeff_means) / coeff_stds
                real_coeffs_norm = (real_coeffs_mat - coeff_means) / coeff_stds
                calib_distances = cdist([gen_coeff_norm], real_coeffs_norm, metric='euclidean')[0]

                total_distances = shape_distances + (calib_weight * calib_distances)
                best_idx = np.argmin(total_distances)
                best_real_coeff = real_coeffs_mat[best_idx]

                f.write(
                    f"{i:03d} | {target_bin:10d} | {shape_distances[best_idx]:14.4f} | {calib_distances[best_idx]:10.4f} | {total_distances[best_idx]:11.4f} | {real_paths[best_idx]}\n")
            else:
                total_distances = shape_distances
                best_idx = np.argmin(total_distances)
                best_real_coeff = None

                f.write(f"{i:03d} | {target_bin:10d} | {total_distances[best_idx]:14.4f} | {real_paths[best_idx]}\n")

            best_real_spec = real_matrix[best_idx]
            closest_file_path = real_paths[best_idx]

            _plot_side_by_side(gen_spec_trimmed, gen_coeff, best_real_spec, best_real_coeff,
                               closest_file_path, total_distances[best_idx], i + 1, retrieval_dir, metric)

    print(f"Retrieval analysis ({metric.upper()}) complete!")


def _plot_side_by_side(gen_spec, gen_coeff, real_spec, real_coeff, closest_file_path, dist, gen_idx, save_dir, metric="euclidean"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    file_name = os.path.basename(closest_file_path)
    L = len(gen_spec)
    channels = np.arange(L)

    # Determine X-Axis (Energy vs Channel)
    if gen_coeff is not None and real_coeff is not None:
        gen_x = np.polyval(gen_coeff[::-1], channels)
        real_x = np.polyval(real_coeff[::-1], channels)
        x_label = "Energy (keV)"
    else:
        gen_x = channels
        real_x = channels
        x_label = "Channel"

    combined = np.concatenate([gen_spec, real_spec])
    positive_vals = combined[combined > 0]

    if len(positive_vals) > 0:
        ymin = max(0.1, float(np.min(positive_vals)) * 0.5)
        ymax = float(np.max(positive_vals)) * 1.5
    else:
        ymin, ymax = 0.1, 1000.0

    max_idx = max(np.nonzero(gen_spec)[0][-1] if np.any(gen_spec) else L,
                  np.nonzero(real_spec)[0][-1] if np.any(real_spec) else L)
    pad = 50
    plot_max_idx = min(max_idx + pad, L - 1)

    # --- Generated Sample (Left Plot) ---
    gen_spec_plot = np.where(gen_spec == 0, 0.1, gen_spec)
    axes[0].plot(gen_x, gen_spec_plot, color='blue', linewidth=1)
    axes[0].set_yscale('log')
    axes[0].set_title(f"Generated Sample {gen_idx:03d}")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Counts (Log Scale)")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)
    axes[0].set_xlim(max(0, gen_x[0]), gen_x[plot_max_idx])
    axes[0].set_ylim(bottom=ymin, top=ymax)

    # --- Closest Real Sample (Right Plot) ---
    real_spec_plot = np.where(real_spec == 0, 0.1, real_spec)
    axes[1].plot(real_x, real_spec_plot, color='green', linewidth=1)
    axes[1].set_yscale('log')
    axes[1].set_title(f"Closest Real ({metric.upper()}): {file_name}\nDistance Score: {dist:.4f}")
    axes[1].set_xlabel(x_label)
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].set_xlim(max(0, real_x[0]), real_x[plot_max_idx])
    axes[1].set_ylim(bottom=ymin, top=ymax)

    # --- Text Annotations (Shifted slightly higher) ---
    # 1. Path centered under the right plot
    axes[1].text(0.5, -0.18, f"Path: {closest_file_path}",
                 horizontalalignment='center', verticalalignment='center',
                 transform=axes[1].transAxes, fontsize=9, color='gray')

    # 2. Both Coefficients stacked vertically and centered at the bottom of the figure
    if gen_coeff is not None and real_coeff is not None:
        gen_str = f"[{', '.join(f'{x:.4f}' for x in gen_coeff)}]"
        real_str = f"[{', '.join(f'{x:.4f}' for x in real_coeff)}]"

        coeff_text = f"Gen Coeffs: {gen_str}\nReal Coeffs: {real_str}"
        fig.text(0.5, 0.05, coeff_text,
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=8, color='gray')

    plt.tight_layout(rect=[0, 0.11, 1, 1])
    plt.savefig(os.path.join(save_dir, f"comparison_{gen_idx:03d}.png"), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    parser = ArgumentParser(description='Find closest real spectra using SIFT or Euclidean metrics')
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the specific inference results folder")
    parser.add_argument("--metric", type=str, choices=["euclidean", "sift"], default="euclidean",
                        help="Matching metric to use ('euclidean' or 'sift')")
    parser.add_argument("--exp", type=str, default="configs/experiments/diffwave_base.yaml",
                        help="Path to experiment config")
    parser.add_argument("--model", type=str, default="configs/models/diffwave.yaml", help="Path to model config")
    parser.add_argument("--calib_weight", type=float, default=0,
                        help="Weight of calibration distance relative to shape distance")
    parser.add_argument("--n_clusters", type=int, default=64,
                        help="Number of SIFT codebook clusters (used only when --metric sift)")

    args = parser.parse_args()

    config = load_config(args.exp, args.model)

    data_dirs = [
        "data/IAEA_HPGe/2024/spectra_txt",
        "data/IAEA_HPGe/2023/spectra_txt",
        "data/Baq_HPGe/spectra_txt"
    ]

    find_and_plot_closest_spectra(
        data_dirs=data_dirs,
        results_dir=args.results_dir,
        config=config,
        metric=args.metric,
        calib_weight=args.calib_weight,
        n_clusters=args.n_clusters
    )