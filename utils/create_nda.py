import argparse
import glob
import os
import random
import shutil
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path to import from utils/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import compute_safe_energies


# ==============================================================================
# Helper Functions & Parsers
# ==============================================================================
def load_coeff_txt(file_path):
    """Parses calibration coefficient files supporting raw numbers or key-value format (e.g., 'c0: 0.123')."""
    coeffs = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            for part in parts:
                clean_part = part.split(":")[-1]  # Strips 'c0:', 'a:', etc.
                try:
                    coeffs.append(float(clean_part))
                except ValueError:
                    continue
    if len(coeffs) != 4:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    return np.array(coeffs, dtype=np.float32)


def load_spectrum_txt(file_path):
    """Parses single spectrum file (.txt) with or without headers."""
    try:
        data = np.loadtxt(
            file_path,
            skiprows=1 if "Channel" in open(file_path).readline() else 0,
        )
        if data.ndim == 2:
            data = data[:, -1]
        return data.astype(np.float32)
    except Exception:
        counts = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if parts and not parts[0].startswith("Channel"):
                    try:
                        counts.append(float(parts[-1]))
                    except ValueError:
                        continue
        return np.array(counts, dtype=np.float32)


def resample_spectrum(spectrum, target_len):
    """Resamples 1D spectrum to target_len if lengths differ."""
    if len(spectrum) == target_len:
        return spectrum
    x_old = np.linspace(0, 1, len(spectrum))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, spectrum).astype(np.float32)


def save_spectrum_txt_with_header(file_path, spectrum):
    """Saves a 1D spectrum to a .txt file formatted as:
    Channel   Counts
    0         0.0000
    1         0.0000
    """
    channels = np.arange(len(spectrum))
    data = np.column_stack((channels, spectrum))
    np.savetxt(
        file_path,
        data,
        fmt=["%d", "%.4f"],
        delimiter="\t",
        header="Channel\tCounts",
        comments="",
    )


# ==============================================================================
# Side-by-Side Plotting Function
# ==============================================================================
def plot_side_by_side_nda(
    corrupted_spec,
    real_spec,
    coeffs,
    source_file_path,
    output_basename,
    sample_idx,
    save_dir,
):
    """Renders side-by-side comparison plots between mutated NDA sample and original real spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    file_name = os.path.basename(source_file_path)
    L = len(corrupted_spec)

    # Calculate x-axis in Energy (keV) if coefficients exist, otherwise Channels
    if coeffs is not None and len(coeffs) == 4:
        x_axis = compute_safe_energies(coeffs, L)
        x_label = "Energy (keV)"
        max_e = np.max(x_axis) if len(x_axis) > 0 else 3000.0
        plot_max_x = float(np.clip(max_e, 30.0, 3000.0))
    else:
        x_axis = np.arange(L)
        x_label = "Channel"
        plot_max_x = L

    # Y-limits based on log scale
    combined = np.concatenate([corrupted_spec, real_spec])
    positive_vals = combined[combined > 0]
    if len(positive_vals) > 0:
        ymin = max(0.1, float(np.min(positive_vals)) * 0.5)
        ymax = float(np.max(positive_vals)) * 2.0
    else:
        ymin, ymax = 0.1, 1000.0

    # --- Mutated NDA Target (Left Plot) ---
    corrupted_plot = np.where(corrupted_spec <= 0, 0.1, corrupted_spec)
    axes[0].plot(x_axis, corrupted_plot, color="blue", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_title(
        f"Mutated NDA Target #{sample_idx:03d}\n{output_basename}",
        fontsize=9,
        fontweight="bold",
    )
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Counts (Log Scale)")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)
    axes[0].set_xlim(0, plot_max_x)
    axes[0].set_ylim(bottom=ymin, top=ymax)

    # --- Original Base Spectrum (Right Plot) ---
    real_plot = np.where(real_spec <= 0, 0.1, real_spec)
    axes[1].plot(x_axis, real_plot, color="green", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_title(
        f"Original Base Spectrum: {file_name}",
        fontsize=10,
        fontweight="bold",
    )
    axes[1].set_xlabel(x_label)
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].set_xlim(0, plot_max_x)
    axes[1].set_ylim(bottom=ymin, top=ymax)

    # --- Text Annotations ---
    axes[1].text(
        0.5,
        -0.18,
        f"Source Path: {source_file_path}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1].transAxes,
        fontsize=8,
        color="gray",
    )

    if coeffs is not None and len(coeffs) == 4:
        coeff_str = f"[{', '.join(f'{x:.4f}' for x in coeffs)}]"
        fig.text(
            0.5,
            0.02,
            f"Calibration Coefficients: {coeff_str}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            color="darkblue",
            fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(os.path.join(save_dir, f"{output_basename}.png"), dpi=300)
    plt.close(fig)


# ==============================================================================
# NDA Core Transformations
# ==============================================================================
def apply_nda_mutations(spectrum, energy_max=3000.0):
    """Applies a randomized combination of unphysical NDA mutations to a 1D spectrum."""
    y_hat = spectrum.copy().astype(np.float32)
    L = len(y_hat)

    params = {}
    suffix_parts = []

    # Choose 1 to 3 random NDA transformations
    mutation_types = ["spikes", "gap", "slope"]
    active_mutations = random.sample(
        mutation_types, k=random.randint(1, 3)
    )

    # 1. Delta Spike Injection
    if "spikes" in active_mutations:
        num_spikes = random.randint(1, 3)
        spikes = []
        spike_str_parts = []

        for _ in range(num_spikes):
            pos_keV = round(random.uniform(1200.0, energy_max - 100.0), 1)
            amp = round(random.uniform(300.0, 5000.0), 1)
            spikes.append({"pos_keV": pos_keV, "amp": amp})

            idx = int(np.clip((pos_keV / energy_max) * L, 0, L - 1))
            y_hat[idx] += amp
            spike_str_parts.append(f"{int(pos_keV)}k-{int(amp)}")

        params["spikes"] = spikes
        suffix_parts.append("spk_" + "_".join(spike_str_parts))

    # 2. Compton Gap Stripping
    if "gap" in active_mutations:
        gap_start = round(random.uniform(400.0, energy_max - 800.0), 1)
        gap_width = round(random.uniform(100.0, 600.0), 1)

        idx_start = int((gap_start / energy_max) * L)
        idx_end = int(((gap_start + gap_width) / energy_max) * L)
        y_hat[idx_start:idx_end] *= 0.01

        params["gap"] = {"start_keV": gap_start, "width_keV": gap_width}
        suffix_parts.append(f"gap{int(gap_start)}-W{int(gap_width)}")

    # 3. Unphysical Slope Reversal
    if "slope" in active_mutations:
        slope_start = round(random.uniform(1000.0, energy_max - 500.0), 1)
        slope_factor = round(random.uniform(0.5, 3.5), 2)

        idx_slope = int((slope_start / energy_max) * L)
        ramp = np.linspace(1.0, 1.0 + slope_factor, L - idx_slope)
        y_hat[idx_slope:] *= ramp

        params["slope"] = {"start_keV": slope_start, "factor": slope_factor}
        suffix_parts.append(f"slp{int(slope_start)}-x{slope_factor:.1f}")

    param_suffix = "_" + "_".join(suffix_parts) if suffix_parts else "_raw"
    return y_hat, params, param_suffix


# ==============================================================================
# Main Generation Pipeline
# ==============================================================================
def generate_nda_dataset(
    source_dirs, output_dir, num_samples, target_len=16384, energy_max=3000.0
):
    out_spectra_dir = os.path.join(output_dir, "spectra_txt")
    out_coeff_dir = os.path.join(output_dir, "coeff_txt")
    out_plots_dir = os.path.join(output_dir, "cal_plots")

    os.makedirs(out_spectra_dir, exist_ok=True)
    os.makedirs(out_coeff_dir, exist_ok=True)
    os.makedirs(out_plots_dir, exist_ok=True)

    source_files = []
    for src_dir in source_dirs:
        spec_txt_path = os.path.join(src_dir, "spectra_txt", "*.txt")
        found = glob.glob(spec_txt_path)
        for f in found:
            source_files.append((f, src_dir))

    if not source_files:
        raise FileNotFoundError(
            f"No spectrum .txt files found in source directories: {source_dirs}"
        )

    print(
        f"Found {len(source_files)} base spectra across {len(source_dirs)} directories."
    )
    print(
        f"Generating {num_samples} NDA mutated samples (Target Length: {target_len}) into: {output_dir}/\n"
    )

    all_corrupted_spectra = []
    all_coeffs = []

    for src_dir in source_dirs:
        potential_bounds = os.path.join(src_dir, "global_calib_bounds.json")
        if os.path.exists(potential_bounds):
            shutil.copy(
                potential_bounds,
                os.path.join(output_dir, "global_calib_bounds.json"),
            )
            break

    for i in range(1, num_samples + 1):
        spec_path, src_dir = random.choice(source_files)
        base_name = os.path.splitext(os.path.basename(spec_path))[0]

        # 1. Load & Resample Spectrum
        real_y = load_spectrum_txt(spec_path)
        real_y = resample_spectrum(real_y, target_len)

        # 2. Apply NDA mutation
        corrupted_y, params, param_suffix = apply_nda_mutations(
            real_y, energy_max=energy_max
        )

        output_basename = f"{base_name}{param_suffix}"

        # 3. Save spectrum .txt with Channel\tCounts header
        out_spec_file = os.path.join(out_spectra_dir, f"{output_basename}.txt")
        save_spectrum_txt_with_header(out_spec_file, corrupted_y)

        # 4. Load calibration coefficient .txt safely
        coeff_src_file = os.path.join(src_dir, "coeff_txt", f"{base_name}.txt")
        out_coeff_file = os.path.join(out_coeff_dir, f"{output_basename}.txt")

        if os.path.exists(coeff_src_file):
            coeffs = load_coeff_txt(coeff_src_file)
            shutil.copy(coeff_src_file, out_coeff_file)
        else:
            coeffs = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            np.savetxt(out_coeff_file, coeffs, fmt="%.6f")

        # 5. Render side-by-side plot directly to cal_plots/
        plot_side_by_side_nda(
            corrupted_spec=corrupted_y,
            real_spec=real_y,
            coeffs=coeffs,
            source_file_path=spec_path,
            output_basename=output_basename,
            sample_idx=i,
            save_dir=out_plots_dir,
        )

        all_corrupted_spectra.append(corrupted_y)
        all_coeffs.append(coeffs)

        if i % 50 == 0 or i == num_samples:
            print(f"  [Progress] Saved {i}/{num_samples} NDA samples & plots...")

    # 6. Save global spectrum and coefficient matrices (.npy) for training
    all_corrupted_matrix = np.array(all_corrupted_spectra, dtype=np.float32)
    all_coeffs_matrix = np.array(all_coeffs, dtype=np.float32)

    matrix_path = os.path.join(output_dir, "spectrum_matrix.npy")
    np.save(matrix_path, all_corrupted_matrix)

    coeff_matrix_path = os.path.join(output_dir, "coeff_matrix.npy")
    np.save(coeff_matrix_path, all_coeffs_matrix)

    print(
        f"\nSuccessfully generated {num_samples} NDA samples in '{output_dir}/'!"
    )
    print(f"  +-- {out_spectra_dir}/")
    print(f"  +-- {out_coeff_dir}/")
    print(f"  +-- {out_plots_dir}/")
    print(f"  +-- {matrix_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch generate Negative Data Augmentation (NDA) samples."
    )
    parser.add_argument(
        "--source_dirs",
        nargs="+",
        default=["data/Baq", "data/IAEA"],
        help="Source dataset folders containing spectra_txt/ and coeff_txt/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/NDA",
        help="Target output directory for NDA dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Total number of mutated NDA samples to generate",
    )
    parser.add_argument(
        "--target_len",
        type=int,
        default=16384,
        help="Target spectrum length in channels",
    )
    parser.add_argument(
        "--energy_max",
        type=float,
        default=3000.0,
        help="Maximum energy in keV",
    )

    args = parser.parse_args()
    generate_nda_dataset(
        source_dirs=args.source_dirs,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        target_len=args.target_len,
        energy_max=args.energy_max,
    )