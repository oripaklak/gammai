import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def load_spectrum(txt_path):
    """Loads spectrum counts from the standardized txt file."""
    # Skip the "Channel\tCounts" header
    data = np.loadtxt(txt_path, skiprows=1)
    if data.ndim == 1:
        return data  # Just in case it's a single column
    return data[:, 1]  # Return only the counts column


def load_coeffs(coeff_path):
    """Loads the 4 calibration parameters from the coeff_txt file."""
    coeffs = np.zeros(4)
    if os.path.exists(coeff_path):
        with open(coeff_path, 'r') as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = float(parts[1].strip())
                    if key == 'c0':
                        coeffs[0] = val
                    elif key == 'c1':
                        coeffs[1] = val
                    elif key == 'c2':
                        coeffs[2] = val
                    elif key == 'c3':
                        coeffs[3] = val
    return coeffs


def process_and_plot_directory(base_dir):
    base_path = Path(base_dir)
    spectra_dir = base_path / "spectra_txt"
    coeff_dir = base_path / "coeff_txt"

    plots_dir = base_path / "plots"
    cal_plots_dir = base_path / "cal_plots"

    plots_dir.mkdir(parents=True, exist_ok=True)
    cal_plots_dir.mkdir(parents=True, exist_ok=True)

    spectra_files = sorted(list(spectra_dir.glob("*.txt")))
    if not spectra_files:
        print(f"No text files found in {spectra_dir}. Skipping.")
        return

    print(f"\nProcessing {len(spectra_files)} files in {base_dir}...")

    for filepath in tqdm(spectra_files, desc="Generating Plots"):
        filename = filepath.stem
        coeff_path = coeff_dir / f"{filename}.txt"

        counts = load_spectrum(filepath)
        coeffs = load_coeffs(coeff_path)

        if len(counts) == 0:
            continue

        original_length = len(counts)
        channels = np.arange(original_length)

        # Calculate Energy: E = c0 + c1*x + c2*x^2 + c3*x^3
        # np.polyval expects coefficients in highest-to-lowest order (c3, c2, c1, c0)
        energy = np.polyval(coeffs[::-1], channels)

        # --- Truncate the signal where a sequence of trailing zeros begins ---
        non_zero_indices = np.nonzero(counts)[0]
        if len(non_zero_indices) > 0:
            truncate_idx = non_zero_indices[-1] + 50  # Add 50 channels for aesthetic padding
            truncate_idx = min(truncate_idx, original_length)
            counts = counts[:truncate_idx]
            channels = channels[:truncate_idx]
            energy = energy[:truncate_idx]

        # --- Dynamic Y-Limits (with the 0.1 floor trick) ---
        positive_vals = counts[counts > 0]
        if len(positive_vals) > 0:
            ymin = max(0.1, float(np.min(positive_vals)) * 0.5)
            ymax = float(np.max(positive_vals)) * 1.5
        else:
            ymin, ymax = 0.1, 1000.0

        # Swap exact 0s for 0.1 so matplotlib draws the baseline correctly
        plot_counts = np.where(counts == 0, 0.1, counts)

        # 1. CHANNEL PLOT (UNCALIBRATED)
        plt.figure(figsize=(10, 5))
        plt.plot(channels, plot_counts, drawstyle='steps-mid', color='blue', linewidth=1)
        plt.xlim(0, len(channels))
        plt.yscale('log')
        plt.ylim(ymin, ymax)
        plt.xlabel('Channel Number')
        plt.ylabel('Counts (Log Scale)')
        plt.title(f'Spectrum: {filename}\n(Original Length: {original_length} Channels)')
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{filename}_plot.png", dpi=150, bbox_inches='tight')
        plt.close()

        # 2. ENERGY PLOT (CALIBRATED)
        plt.figure(figsize=(10, 5))
        plt.plot(energy, plot_counts, drawstyle='steps-mid', color='blue', linewidth=1)
        plt.xlim(np.min(energy), np.max(energy))
        plt.yscale('log')
        plt.ylim(ymin, ymax)
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts (Log Scale)')
        plt.title(f'Calibrated Spectrum: {filename}\n(Energy Range: {np.min(energy):.1f} - {np.max(energy):.1f} keV)')
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(cal_plots_dir / f"{filename}_cal_plot.png", dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Add your parent dataset directories here
    dataset_dirs = [
        "data/IAEA_HPGe/2023",
        "data/IAEA_HPGe/2024",
        "data/Baq_HPGe"
    ]

    for d in dataset_dirs:
        process_and_plot_directory(d)

    print("\nAll plotting complete!")