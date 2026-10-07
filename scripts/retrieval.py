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

# Standard global energy grid (keV) for uniform cross-calibration alignment
COMMON_ENERGY_GRID = np.linspace(0, 2500, 5000)


def resample_spectrum_to_energy_grid(spectrum: np.ndarray, coeffs: np.ndarray,
                                     energy_grid: np.ndarray = COMMON_ENERGY_GRID) -> np.ndarray:
    """
    Resamples a 1D gamma-ray spectrum from channel space to a standard energy grid (keV).
    Applies a log1p transformation to stabilize variance across count dynamic ranges.
    """
    L = len(spectrum)
    channels = np.arange(L)

    if coeffs is not None:
        # Evaluate energy values using polynomial calibration coefficients: c3*x^3 + c2*x^2 + c1*x + c0
        energies = np.maximum(np.polyval(coeffs[::-1], channels), 0.0)
    else:
        energies = channels.astype(np.float32)

    # Interpolate raw spectrum counts onto the common energy grid
    resampled_counts = np.interp(energy_grid, energies, spectrum, left=0.0, right=0.0)

    # Log-scale transformation for structural shape matching
    return np.log1p(resampled_counts)


def compute_chi_square_distances(query_hist: np.ndarray, dataset_hists: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Vectorized Chi-Square distance between a single SIFT query histogram (K,)
    and a dataset of histograms (N, K).
    """
    diff = query_hist - dataset_hists
    denom = query_hist + dataset_hists + eps
    return 0.5 * np.sum((diff ** 2) / denom, axis=1)


def load_grouped_real_data(data_dirs, is_calib=False):
    """
    Reads and parses real spectrum .txt files and associated calibration coefficients.
    """
    bins = [4096, 8192, 16384]
    grouped_data = {b: [] for b in bins}
    grouped_coeffs = {b: [] for b in bins} if is_calib else None
    grouped_paths = {b: [] for b in bins}

    print(f"Reading real dataset spectra (Calibration Mode: {is_calib})...")
    found_any = False

    for d in data_dirs:
        folder_path = Path(d)
        if not folder_path.exists():
            continue

        # Search for calibration coefficient folder across common relative locations
        coeff_dir = None
        if is_calib:
            for cand in [folder_path.parent / "coeff_txt", folder_path / "coeff_txt", folder_path.parent / "coeff"]:
                if cand.exists():
                    coeff_dir = cand
                    break

        for file_path in sorted(folder_path.glob("*.txt")):
            if is_calib:
                if coeff_dir is None:
                    continue
                coeff_path = coeff_dir / file_path.name
                if not coeff_path.exists():
                    continue

            try:
                # 1. Load spectrum counts
                spectrum = np.loadtxt(file_path, dtype=np.float32, skiprows=1, usecols=1)
                L = len(spectrum)
                closest_bin = min(bins, key=lambda x: abs(x - L))

                if L < closest_bin:
                    spectrum = np.pad(spectrum, (0, closest_bin - L), 'constant')
                elif L > closest_bin:
                    spectrum = spectrum[:closest_bin]

                # 2. Load calibration coefficients
                if is_calib:
                    coeffs = np.zeros(4, dtype=np.float32)
                    with open(coeff_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.split(":")
                            if len(parts) == 2:
                                key, val = parts[0].strip(), float(parts[1].strip())
                                if key in ['c0', 'c1', 'c2', 'c3']:
                                    idx = int(key[1])
                                    coeffs[idx] = val
                    grouped_coeffs[closest_bin].append(coeffs)

                grouped_data[closest_bin].append(spectrum)
                grouped_paths[closest_bin].append(str(file_path))
                found_any = True
            except Exception:
                pass

    for b in bins:
        if len(grouped_data[b]) > 0:
            grouped_data[b] = np.array(grouped_data[b], dtype=np.float32)
            if is_calib:
                grouped_coeffs[b] = np.array(grouped_coeffs[b], dtype=np.float32)
            print(f"  -> Loaded {len(grouped_data[b])} real spectra for bin {b}")
        else:
            grouped_data[b] = np.empty((0, b), dtype=np.float32)
            if is_calib:
                grouped_coeffs[b] = np.empty((0, 4), dtype=np.float32)

    if not found_any:
        raise FileNotFoundError(
            f"No valid real spectra .txt files found in provided directories:\n"
            + "\n".join(f"  - {d}" for d in data_dirs)
        )

    return grouped_data, grouped_coeffs, grouped_paths


def find_and_plot_closest_spectra(data_dirs, results_dir, config, metric="cosine", calib_weight=0.0, n_clusters=64):
    """
    Main retrieval engine: compares generated spectra against the real dataset on an energy-aligned grid.
    """
    is_calib = config.get('calibration', {}).get('enabled', False)

    # 1. Load generated samples
    gen_files = [f for f in os.listdir(results_dir) if f.startswith("all_") and f.endswith("samples.npy")]
    if not gen_files:
        raise FileNotFoundError(f"No generated samples .npy file found in {results_dir}")
    gen_data = np.load(os.path.join(results_dir, gen_files[0])).astype(np.float32)
    num_samples = gen_data.shape[0]

    # 2. Load generated calibration coefficients if enabled
    gen_coeffs = None
    if is_calib:
        coeff_files = [f for f in os.listdir(results_dir) if f.startswith("all_") and f.endswith("coeffs.npy")]
        if not coeff_files:
            raise FileNotFoundError(f"No generated coeffs .npy file found in {results_dir}")
        gen_coeffs = np.load(os.path.join(results_dir, coeff_files[0])).astype(np.float32)

    # 3. Load real dataset
    real_data_groups, real_coeffs_groups, real_paths_groups = load_grouped_real_data(data_dirs, is_calib)

    # 4. Initialize SIFT model if SIFT metric is chosen
    bow_model = None
    real_sift_groups = {}
    if metric == "sift":
        print("\nInitializing 1D SIFT Extractor & Codebook on energy-aligned signals...")
        dataset_name = config.get("paths", {}).get("dataset_name", "dataset")
        codebook_dir = Path("data/codebooks")
        codebook_dir.mkdir(parents=True, exist_ok=True)
        codebook_path = codebook_dir / f"sift_codebook_{dataset_name}_k{n_clusters}.pkl"

        sift_extractor = TemporalSIFT(tau_step=15, num_scales=6, n_b=4, a=2)
        bow_model = SIFTBagOfWords(extractor=sift_extractor, n_clusters=n_clusters)

        # Resample all real spectra to the common energy grid for SIFT training
        all_real_aligned = []
        for b in real_data_groups:
            if len(real_data_groups[b]) > 0:
                r_coeffs = real_coeffs_groups[b] if is_calib else [None] * len(real_data_groups[b])
                for spec, c_val in zip(real_data_groups[b], r_coeffs):
                    aligned = resample_spectrum_to_energy_grid(spec, c_val)
                    all_real_aligned.append(aligned)

        if codebook_path.exists():
            print(f"Loading SIFT Codebook from '{codebook_path}'...")
            bow_model.load_codebook(str(codebook_path))
        else:
            print(f"Training new SIFT Codebook on {len(all_real_aligned)} aligned real spectra...")
            bow_model.fit_codebook(all_real_aligned)
            bow_model.save_codebook(str(codebook_path))

        print("Encoding real energy-aligned spectra to SIFT histograms...")
        for b in real_data_groups:
            if len(real_data_groups[b]) > 0:
                r_coeffs = real_coeffs_groups[b] if is_calib else [None] * len(real_data_groups[b])
                hist_list = []
                for spec, c_val in zip(real_data_groups[b], r_coeffs):
                    aligned = resample_spectrum_to_energy_grid(spec, c_val)
                    hist_list.append(bow_model.signal_to_histogram(aligned))
                real_sift_groups[b] = np.array(hist_list)
            else:
                real_sift_groups[b] = np.empty((0, n_clusters))

    # 5. Output setup
    mode_str = "calibrated" if is_calib else "uncalibrated"
    retrieval_dir = os.path.join(results_dir, f"retrieval_comparisons_{metric}_{mode_str}_calib_weight_{calib_weight}")
    os.makedirs(retrieval_dir, exist_ok=True)

    dist_label = f"{metric.upper()}_Dist"

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
                non_empty_bins = [b for b in bins if len(real_data_groups[b]) > 0]
                if not non_empty_bins:
                    raise RuntimeError("No real spectra available in dataset.")
                target_bin = non_empty_bins[0]

            real_matrix = real_data_groups[target_bin]
            real_coeffs_mat = real_coeffs_groups[target_bin] if is_calib else [None] * len(real_matrix)
            real_paths = real_paths_groups[target_bin]

            # --- Align generated and real spectra on standard Energy Grid (keV) ---
            gen_energy_aligned = resample_spectrum_to_energy_grid(gen_spec, gen_coeff)

            real_energy_aligned = np.array([
                resample_spectrum_to_energy_grid(spec, c_val)
                for spec, c_val in zip(real_matrix, real_coeffs_mat)
            ])

            # Shape distance computation on energy-aligned arrays
            if metric == "sift":
                gen_sift_hist = bow_model.signal_to_histogram(gen_energy_aligned)
                shape_distances = compute_chi_square_distances(gen_sift_hist, real_sift_groups[target_bin])
            elif metric in ["cosine", "euclidean"]:
                shape_distances = cdist([gen_energy_aligned], real_energy_aligned, metric=metric)[0]
            else:
                shape_distances = cdist([gen_energy_aligned], real_energy_aligned, metric="euclidean")[0]

            # Calibration coefficient distance computation (if enabled)
            if is_calib and calib_weight > 0:
                real_coeffs_array = np.array(real_coeffs_mat)
                coeff_means = np.mean(real_coeffs_array, axis=0)
                coeff_stds = np.std(real_coeffs_array, axis=0) + 1e-8
                gen_coeff_norm = (gen_coeff - coeff_means) / coeff_stds
                real_coeffs_norm = (real_coeffs_array - coeff_means) / coeff_stds
                calib_distances = cdist([gen_coeff_norm], real_coeffs_norm, metric='euclidean')[0]

                total_distances = shape_distances + (calib_weight * calib_distances)
                best_idx = np.argmin(total_distances)
                best_real_coeff = real_coeffs_array[best_idx]

                f.write(
                    f"{i:03d} | {target_bin:10d} | {shape_distances[best_idx]:14.4f} | "
                    f"{calib_distances[best_idx]:10.4f} | {total_distances[best_idx]:11.4f} | {real_paths[best_idx]}\n"
                )
            else:
                total_distances = shape_distances
                best_idx = np.argmin(total_distances)
                best_real_coeff = real_coeffs_mat[best_idx] if is_calib else None

                f.write(f"{i:03d} | {target_bin:10d} | {total_distances[best_idx]:14.4f} | {real_paths[best_idx]}\n")

            best_real_spec = real_matrix[best_idx]
            closest_file_path = real_paths[best_idx]

            _plot_side_by_side(
                gen_spec, gen_coeff, best_real_spec, best_real_coeff,
                closest_file_path, total_distances[best_idx], i + 1, retrieval_dir, metric
            )

    print(f"Retrieval analysis ({metric.upper()}) complete! Results saved in: {retrieval_dir}")


def _plot_side_by_side(gen_spec, gen_coeff, real_spec, real_coeff, closest_file_path, dist, gen_idx, save_dir,
                       metric="cosine"):
    """
    Renders side-by-side comparison plots between generated sample and retrieved real spectrum.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    file_name = os.path.basename(closest_file_path)

    # Calculate x-axis in Energy (keV) if coefficients exist, otherwise Channels
    if gen_coeff is not None:
        gen_x = np.maximum(np.polyval(gen_coeff[::-1], np.arange(len(gen_spec))), 0.0)
        x_label = "Energy (keV)"
    else:
        gen_x = np.arange(len(gen_spec))
        x_label = "Channel"

    if real_coeff is not None:
        real_x = np.maximum(np.polyval(real_coeff[::-1], np.arange(len(real_spec))), 0.0)
    else:
        real_x = np.arange(len(real_spec))

    # Y-limits based on log scale
    combined = np.concatenate([gen_spec, real_spec])
    positive_vals = combined[combined > 0]
    if len(positive_vals) > 0:
        ymin = max(0.1, float(np.min(positive_vals)) * 0.5)
        ymax = float(np.max(positive_vals)) * 1.5
    else:
        ymin, ymax = 0.1, 1000.0

    # Dynamic X-limit cropping based on last non-zero channel
    gen_non_zeros = np.nonzero(gen_spec)[0]
    real_non_zeros = np.nonzero(real_spec)[0]
    max_idx_gen = gen_non_zeros[-1] if len(gen_non_zeros) > 0 else len(gen_spec) - 1
    max_idx_real = real_non_zeros[-1] if len(real_non_zeros) > 0 else len(real_spec) - 1

    plot_max_x_gen = gen_x[min(max_idx_gen + 50, len(gen_spec) - 1)]
    plot_max_x_real = real_x[min(max_idx_real + 50, len(real_spec) - 1)]

    # --- Generated Sample (Left Plot) ---
    gen_spec_plot = np.where(gen_spec <= 0, 0.1, gen_spec)
    axes[0].plot(gen_x, gen_spec_plot, color='blue', linewidth=1)
    axes[0].set_yscale('log')
    axes[0].set_title(f"Generated Sample {gen_idx:03d}")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Counts (Log Scale)")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)
    axes[0].set_xlim(0, plot_max_x_gen)
    axes[0].set_ylim(bottom=ymin, top=ymax)

    # --- Closest Real Sample (Right Plot) ---
    real_spec_plot = np.where(real_spec <= 0, 0.1, real_spec)
    axes[1].plot(real_x, real_spec_plot, color='green', linewidth=1)
    axes[1].set_yscale('log')
    axes[1].set_title(f"Closest Real ({metric.upper()}): {file_name}\nDistance Score: {dist:.4f}")
    axes[1].set_xlabel(x_label)
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].set_xlim(0, plot_max_x_real)
    axes[1].set_ylim(bottom=ymin, top=ymax)

    # --- Text Annotations ---
    axes[1].text(0.5, -0.18, f"Path: {closest_file_path}",
                 horizontalalignment='center', verticalalignment='center',
                 transform=axes[1].transAxes, fontsize=9, color='gray')

    if gen_coeff is not None and real_coeff is not None:
        gen_str = f"[{', '.join(f'{x:.4f}' for x in gen_coeff)}]"
        real_str = f"[{', '.join(f'{x:.4f}' for x in real_coeff)}]"
        coeff_text = f"Gen Coeffs: {gen_str}\nReal Coeffs: {real_str}"
        fig.text(0.5, 0.03, coeff_text,
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=8, color='gray')

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.savefig(os.path.join(save_dir, f"comparison_{gen_idx:03d}.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(
        description='Find closest real spectra using Energy-Aligned SIFT, Cosine, or Euclidean metrics')
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the specific inference results folder")
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean", "sift"], default="cosine",
                        help="Matching metric to use ('cosine', 'euclidean', or 'sift')")
    parser.add_argument("--exp", type=str, default="configs/experiments/diffwave_Baq.yaml",
                        help="Path to experiment config")
    parser.add_argument("--model", type=str, default="configs/models/diffwave.yaml", help="Path to model config")
    parser.add_argument("--calib_weight", type=float, default=0.002,
                        help="Weight of calibration coefficient distance relative to spectral shape distance")
    parser.add_argument("--n_clusters", type=int, default=64,
                        help="Number of SIFT codebook clusters (used only when --metric sift)")

    args = parser.parse_args()

    config = load_config(args.exp, args.model)

    # Automatically build target real dataset directories based on experiment configuration
    base_dir = config.get("paths", {}).get("base_dir", "data")
    dataset_name = config.get("paths", {}).get("dataset_name", "Baq")
    dataset_dir = os.path.join(base_dir, dataset_name)

    data_dirs = [
        "data/Baq/spectra_txt",
        # "data/IAEA/2023/spectra_txt",
        # "data/IAEA/2024/spectra_txt",
    ]

    find_and_plot_closest_spectra(
        data_dirs=data_dirs,
        results_dir=args.results_dir,
        config=config,
        metric=args.metric,
        calib_weight=args.calib_weight,
        n_clusters=args.n_clusters
    )