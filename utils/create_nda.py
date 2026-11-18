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

try:
    from utils.helpers import compute_energies
except ImportError:
    def compute_energies(coeffs, L):
        ch = np.arange(L, dtype=np.float32)
        return coeffs[0] + coeffs[1] * ch + coeffs[2] * (ch ** 2) + coeffs[3] * (ch ** 3)


# ==============================================================================
# Helper Functions & Parsers
# ==============================================================================
def load_coeff_txt(file_path):
    """Parses calibration coefficient files. Returns None if invalid or dummy."""
    if not file_path or not os.path.exists(file_path):
        return None
    coeffs = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            for part in parts:
                clean_part = part.split(":")[-1]
                try:
                    coeffs.append(float(clean_part))
                except ValueError:
                    continue
    if len(coeffs) != 4 or coeffs == [0.0, 1.0, 0.0, 0.0]:
        return None
    return np.array(coeffs, dtype=np.float32)


def load_spectrum_txt(file_path):
    """Parses single spectrum file (.txt) with or without headers strictly."""
    counts = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].startswith("Channel"):
                continue
            val_str = parts[-1].replace(".", "", 1).replace("-", "", 1)
            if not val_str.isdigit():
                continue
            try:
                counts.append(float(parts[-1]))
            except ValueError:
                continue
    return np.array(counts, dtype=np.float32)


def resample_spectrum(spectrum, target_len):
    """Resamples 1D spectrum to target_len if lengths differ."""
    if len(spectrum) == target_len or len(spectrum) == 0:
        return spectrum
    x_old = np.linspace(0, 1, len(spectrum))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, spectrum).astype(np.float32)


def save_spectrum_txt_with_header(file_path, spectrum):
    """Saves a 1D spectrum to a .txt file formatted as Channel \\t Counts."""
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


def get_local_environment(spectrum, idx, window_bins=25):
    """Calculates the local continuum/background level around channel idx."""
    L = len(spectrum)
    start = max(0, idx - window_bins)
    end = min(L, idx + window_bins + 1)
    window = spectrum[start:end]
    positive_bins = window[window > 0]

    if len(positive_bins) >= 3:
        return float(np.median(positive_bins))

    all_positive = spectrum[spectrum > 0]
    if len(all_positive) > 0:
        return float(np.percentile(all_positive, 5))
    return 1.0


def add_gaussian_peak(spectrum, idx, added_amplitude, fwhm_bins=6):
    """Spreads the peak over neighboring channels using a Gaussian profile."""
    L = len(spectrum)
    sigma = max(0.8, fwhm_bins / 2.355)
    span = int(max(3, 3 * sigma))
    low = max(0, idx - span)
    high = min(L, idx + span + 1)
    chans = np.arange(low, high)
    kernel = added_amplitude * np.exp(-0.5 * ((chans - idx) / sigma) ** 2)
    spectrum[low:high] += kernel


# ==============================================================================
# Side-by-Side Plotting Function with Injected Peak Tracking
# ==============================================================================
def plot_side_by_side_nda(
        corrupted_spec,
        real_spec,
        coeffs,
        source_file_path,
        output_basename,
        sample_idx,
        save_dir,
        peaks_info=None,
):
    """Renders side-by-side comparison plots and an itemized table of injected peaks."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.8))
    file_name = os.path.basename(source_file_path)
    abs_source_path = os.path.abspath(source_file_path)
    L = len(corrupted_spec)

    # Calculate x-axis in Energy (keV) if valid calibration exists, otherwise Channels
    if coeffs is not None:
        x_axis = compute_energies(coeffs, L)
        x_label = "Energy (keV)"
        max_e = np.max(x_axis) if len(x_axis) > 0 else 3000.0
        plot_max_x = float(np.clip(max_e, 30.0, 5000.0))
        cal_str = f"[{', '.join(f'{x:.4f}' for x in coeffs)}]"
    else:
        x_axis = np.arange(L, dtype=np.float32)
        x_label = "Channel"
        plot_max_x = float(L)
        cal_str = "None (Uncalibrated Channels)"

    # Determine dynamic Y-limits based on non-zero data
    valid_vals = np.concatenate([corrupted_spec[corrupted_spec > 0], real_spec[real_spec > 0]])
    ymin = max(1e-4, float(np.min(valid_vals)) * 0.5) if len(valid_vals) > 0 else 0.1
    ymax = float(np.max(valid_vals)) * 3.0 if len(valid_vals) > 0 else 1000.0

    # --- Mutated NDA Target (Left Plot) ---
    corrupted_plot = np.where(corrupted_spec <= 0, ymin, corrupted_spec)
    axes[0].plot(x_axis, corrupted_plot, color="#1f77b4", linewidth=1.1, label="Mutated Signal")
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

    # Plot peak markers on the left plot
    if peaks_info:
        for p in peaks_info:
            idx = p["channel"]
            pos_x = x_axis[idx]
            peak_y = corrupted_plot[idx]

            is_end = p["is_end"]
            is_gauss = (p["type"] == "Gaussian")

            color = "#d62728" if is_end else ("#9467bd" if is_gauss else "#e377c2")
            marker = "o" if is_gauss else "^"

            axes[0].scatter(pos_x, peak_y, color=color, s=28, marker=marker, zorder=6, alpha=0.85)

        axes[0].scatter([], [], color="#9467bd", marker="o", label="Gaussian (Body)")
        axes[0].scatter([], [], color="#e377c2", marker="^", label="Delta Spike (Body)")
        axes[0].scatter([], [], color="#d62728", marker="o", label="End Peak (High-E Tail)")
        axes[0].legend(loc="upper right", fontsize=7.5)

    # --- Original Base Spectrum (Right Plot) ---
    real_plot = np.where(real_spec <= 0, ymin, real_spec)
    axes[1].plot(x_axis, real_plot, color="#2ca02c", linewidth=1.1, label="Original Spectrum")
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
    axes[1].legend(loc="upper right", fontsize=7.5)

    # Source info
    axes[1].text(
        0.5,
        -0.15,
        f"Source Full Path: {abs_source_path}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1].transAxes,
        fontsize=7.5,
        color="#444444",
        family="monospace",
    )

    fig.text(
        0.05,
        0.18,
        f"Calibration: {cal_str}",
        fontsize=8,
        color="darkblue",
        fontweight="bold",
    )

    # --- Accurate Peak Tracking Table ---
    if peaks_info:
        fig.text(0.05, 0.15, f"Injected Peaks ({len(peaks_info)} total):", fontsize=8.5, fontweight="bold")
        cols = 4
        rows_per_col = int(np.ceil(len(peaks_info) / cols))
        unit = "keV" if coeffs is not None else "ch"

        for i, p in enumerate(peaks_info):
            col_idx = i // rows_per_col
            row_idx = i % rows_per_col
            x_pos = 0.05 + col_idx * 0.235
            y_pos = 0.12 - (row_idx * 0.024)

            p_type = "Gauss" if p["type"] == "Gaussian" else "Delta"
            loc_tag = "END " if p["is_end"] else "Body"
            entry_str = f"#{i + 1:02d}: {p['pos']:>6.1f} {unit} [{p_type:<5}|{loc_tag}] ×{p['mult']:.1f}"

            fig.text(x_pos, y_pos, entry_str, fontsize=7.5, family="monospace", color="#222222")

    plt.tight_layout(rect=[0, 0.20, 1, 1])
    plt.savefig(os.path.join(save_dir, f"{output_basename}.png"), dpi=300)
    plt.close(fig)


# ==============================================================================
# NDA Core Transformations
# ==============================================================================
def apply_nda_mutations(spectrum, coeffs=None):
    """Applies a mixture of 15-22 Delta/Gaussian peaks with minimum multiplier of 4.5."""
    y_hat = spectrum.copy().astype(np.float32)
    L = len(y_hat)

    # Compute actual calibrated axis so positions match exactly
    if coeffs is not None:
        actual_x_axis = compute_energies(coeffs, L)
    else:
        actual_x_axis = np.arange(L, dtype=np.float32)

    params = {}
    suffix_parts = []

    # 1. Total peaks: guaranteed 15 to 22
    num_peaks = random.randint(15, 22)
    peaks_record = []

    # 2. End peaks: 3 to 5 peaks strictly in the tail
    num_end_peaks = random.randint(3, 5)
    num_body_peaks = num_peaks - num_end_peaks

    # Body channels: 5% to 88% of spectrum length
    body_indices = [random.randint(int(L * 0.05), int(L * 0.88)) for _ in range(num_body_peaks)]

    # End peaks: strictly in the 93% to 99.5% range
    end_indices = [random.randint(int(L * 0.93), int(L * 0.995)) for _ in range(num_end_peaks)]

    tagged_indices = [(idx, False) for idx in body_indices] + [(idx, True) for idx in end_indices]
    tagged_indices.sort(key=lambda x: x[0])

    num_deltas = 0
    num_gaussians = 0

    for idx, is_end in tagged_indices:
        local_bg = get_local_environment(y_hat, idx)

        # Multiplier set to at least 4.5 (range: 4.5 to 14.0)
        multiplier = round(random.uniform(4.5, 14.0), 2)
        added_amp = max(local_bg * multiplier, 2.0)

        # 50/50 mixture of Delta Spikes and Gaussian peaks
        peak_type = random.choice(["Delta", "Gaussian"])

        if peak_type == "Gaussian":
            num_gaussians += 1
            fwhm = random.randint(5, 12)
            add_gaussian_peak(y_hat, idx, added_amp, fwhm_bins=fwhm)
        else:
            num_deltas += 1
            fwhm = 1
            y_hat[idx] += added_amp

        exact_energy_pos = round(float(actual_x_axis[idx]), 1)
        peaks_record.append({
            "channel": idx,
            "pos": exact_energy_pos,
            "type": peak_type,
            "fwhm": fwhm,
            "is_end": is_end,
            "mult": multiplier,
            "amp": float(added_amp),
        })

    params["peaks"] = peaks_record
    suffix_parts.append(f"{num_peaks}pks({num_gaussians}G_{num_deltas}D_{num_end_peaks}End)")

    # ==========================================================================
    # 2. Compton Gap Stripping (Commented out per request)
    # ==========================================================================
    # if random.random() < 0.5:
    #     gap_start = round(random.uniform(400.0, energy_max - 800.0), 1)
    #     gap_width = round(random.uniform(100.0, 600.0), 1)
    #     idx_start = int((gap_start / energy_max) * L)
    #     idx_end = int(((gap_start + gap_width) / energy_max) * L)
    #     y_hat[idx_start:idx_end] *= 0.01
    #     params["gap"] = {"start_keV": gap_start, "width_keV": gap_width}
    #     suffix_parts.append(f"gap{int(gap_start)}-W{int(gap_width)}")

    # 3. Slope Distortion (Multiplier in range 0.0 - 1.0)
    if random.random() < 0.5:
        idx_slope = random.randint(int(L * 0.4), int(L * 0.8))
        slope_factor = round(random.uniform(0.0, 1.0), 2)

        ramp = np.linspace(1.0, slope_factor, L - idx_slope)
        y_hat[idx_slope:] *= ramp

        approx_start = round(float(actual_x_axis[idx_slope]), 1)
        params["slope"] = {"start": approx_start, "factor": slope_factor}
        suffix_parts.append(f"slp-x{slope_factor:.2f}")

    param_suffix = "_" + "_".join(suffix_parts) if suffix_parts else "_raw"
    return y_hat, params, param_suffix


# ==============================================================================
# Main Generation Pipeline
# ==============================================================================
def generate_nda_dataset(
        source_dirs, output_dir, num_samples, target_len=16384
):
    out_spectra_dir = os.path.join(output_dir, "spectra_txt")
    out_coeff_dir = os.path.join(output_dir, "coeff_txt")
    out_plots_dir = os.path.join(output_dir, "cal_plots")

    os.makedirs(out_spectra_dir, exist_ok=True)
    os.makedirs(out_coeff_dir, exist_ok=True)
    os.makedirs(out_plots_dir, exist_ok=True)

    source_files = []
    for src_dir in source_dirs:
        if not os.path.isdir(src_dir):
            continue
        patterns = [
            os.path.join(src_dir, "spectra_txt", "*.txt"),
            os.path.join(src_dir, "**", "spectra_txt", "*.txt"),
        ]
        for pat in patterns:
            for f in glob.glob(pat, recursive=True):
                if os.path.isfile(f) and f.endswith(".txt"):
                    source_files.append((os.path.abspath(f), os.path.abspath(src_dir)))

    source_files = list(set(source_files))

    if not source_files:
        raise FileNotFoundError(
            f"No .txt spectrum files found strictly inside 'spectra_txt/' folders in: {source_dirs}"
        )

    print(f"Found {len(source_files)} base spectra across {len(source_dirs)} directories.")
    print(
        f"Generating {num_samples} NDA samples (Multiplier: 4.5-14.0x, Target Length: {target_len}) into: {output_dir}/\n")

    all_corrupted_spectra = []
    all_coeffs = []

    for src_dir in source_dirs:
        potential_bounds = os.path.join(src_dir, "global_calib_bounds.json")
        if os.path.exists(potential_bounds):
            shutil.copy(potential_bounds, os.path.join(output_dir, "global_calib_bounds.json"))
            break

    for i in range(1, num_samples + 1):
        spec_path, src_dir = random.choice(source_files)
        base_name = os.path.splitext(os.path.basename(spec_path))[0]

        # 1. Calibration loading
        spec_dir = os.path.dirname(spec_path)
        parent_dir = os.path.dirname(spec_dir)
        possible_coeff_paths = [
            os.path.join(parent_dir, "coeff_txt", f"{base_name}.txt"),
            os.path.join(src_dir, "coeff_txt", f"{base_name}.txt"),
        ]
        coeff_src_file = next((p for p in possible_coeff_paths if os.path.exists(p)), None)
        coeffs = load_coeff_txt(coeff_src_file)

        # 2. Resample spectrum
        real_y = load_spectrum_txt(spec_path)
        real_y = resample_spectrum(real_y, target_len)

        # 3. Apply mutations (Multiplier >= 4.5)
        corrupted_y, params, param_suffix = apply_nda_mutations(
            real_y, coeffs=coeffs
        )

        output_basename = f"{base_name}{param_suffix}"

        # 4. Save spectrum .txt
        out_spec_file = os.path.join(out_spectra_dir, f"{output_basename}.txt")
        save_spectrum_txt_with_header(out_spec_file, corrupted_y)

        # 5. Save calibration .txt
        out_coeff_file = os.path.join(out_coeff_dir, f"{output_basename}.txt")
        if coeffs is not None and coeff_src_file and os.path.exists(coeff_src_file):
            shutil.copy(coeff_src_file, out_coeff_file)
        else:
            dummy_coeffs = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            np.savetxt(out_coeff_file, dummy_coeffs, fmt="%.6f")

        # 6. Render side-by-side plot with peak tracking markers and table
        plot_side_by_side_nda(
            corrupted_spec=corrupted_y,
            real_spec=real_y,
            coeffs=coeffs,
            source_file_path=spec_path,
            output_basename=output_basename,
            sample_idx=i,
            save_dir=out_plots_dir,
            peaks_info=params.get("peaks", []),
        )

        all_corrupted_spectra.append(corrupted_y)
        all_coeffs.append(coeffs if coeffs is not None else np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

        if i % 25 == 0 or i == num_samples:
            print(f"  [Progress] Saved {i}/{num_samples} NDA samples & plots...")

    # 7. Save matrices (.npy)
    all_corrupted_matrix = np.array(all_corrupted_spectra, dtype=np.float32)
    all_coeffs_matrix = np.array(all_coeffs, dtype=np.float32)

    matrix_path = os.path.join(output_dir, "spectrum_matrix.npy")
    np.save(matrix_path, all_corrupted_matrix)

    coeff_matrix_path = os.path.join(output_dir, "coeff_matrix.npy")
    np.save(coeff_matrix_path, all_coeffs_matrix)

    print(f"\nSuccessfully generated {num_samples} NDA samples in '{output_dir}/'!")
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
        default=20,
        help="Total number of mutated NDA samples to generate",
    )
    parser.add_argument(
        "--target_len",
        type=int,
        default=16384,
        help="Target spectrum length in channels",
    )

    args = parser.parse_args()
    generate_nda_dataset(
        source_dirs=args.source_dirs,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        target_len=args.target_len,
    )