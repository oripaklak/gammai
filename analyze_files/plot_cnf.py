import os

# Set legacy promotion state before importing numpy/matplotlib
os.environ['NPY_PROMOTION_STATE'] = 'legacy'

import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from read_cnf import read_cnf_file

# ==========================================
# SETUP AND DATA LOADING
# ==========================================
dataset_dir = "data/Baq_HPGe/cnf"
output_dir = "data/Baq_HPGe/plots"
cal_output_dir = "data/Baq_HPGe/cal_plots"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(cal_output_dir, exist_ok=True)

all_cnf_files = glob.glob(os.path.join(dataset_dir, "*.[cC][nN][fF]"))
all_cnf_files.sort()

print(f"Found {len(all_cnf_files)} CNF files. Generating uncalibrated and calibrated plots...")

# ==========================================
# PLOTTING LOOP
# ==========================================
for filepath in tqdm(all_cnf_files, desc="Saving CNF Plots"):
    try:
        cnf_data = read_cnf_file(filepath)
        counts = cnf_data.get('Channels data')

        if counts is None or len(counts) == 0:
            print(f"\nWarning: Could not find 'Channels data' in {filepath}. Skipping.")
            continue

        counts = np.array(counts)
        original_length = len(counts)

        # --- Get or calculate Energy values ---
        energy_unit = cnf_data.get('Energy unit', 'keV')
        energy = cnf_data.get('Energy')

        if energy is None:
            # Fallback calculation using Energy coefficients [c0, c1, c2, c3]
            coeffs = cnf_data.get('Energy coefficients')
            channels = cnf_data.get('Channels', np.arange(1, original_length + 1))
            if coeffs is not None:
                energy = np.polyval(coeffs[::-1], channels)
            else:
                energy = channels  # Fallback to channel indices if no calibration exists

        # --- Truncate the signal where a sequence of zeros begins ---
        window_size = 50  # Number of consecutive zeros that defines a "sequence"
        is_zero = (counts == 0).astype(int)

        zero_runs = np.convolve(is_zero, np.ones(window_size, dtype=int), mode='valid')
        seq_indices = np.where(zero_runs == window_size)[0]

        if len(seq_indices) > 0:
            truncate_idx = seq_indices[0]
            counts = counts[:truncate_idx]
            energy = energy[:truncate_idx]
        else:
            non_zero_indices = np.nonzero(counts)[0]
            if len(non_zero_indices) > 0:
                truncate_idx = non_zero_indices[-1] + 1
                counts = counts[:truncate_idx]
                energy = energy[:truncate_idx]

        base_name = os.path.basename(filepath)
        name_without_ext = os.path.splitext(base_name)[0]

        min_val = max(1, np.min(counts))
        max_val = np.max(counts)

        # ==========================================
        # 1. CHANNEL PLOT (UNCALIBRATED)
        # ==========================================
        plt.figure(figsize=(10, 5))
        plt.plot(counts, drawstyle='steps-mid', color='blue', linewidth=1)
        plt.xlim(0, len(counts))
        plt.yscale('log')
        plt.ylim(min_val, max_val * 1.2)
        plt.xlabel('Channel Number')
        plt.ylabel('Counts (Log Scale)')
        plt.title(f'Spectrum: {base_name}\n(Original Length: {original_length} Channels)')
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()

        output_filename = os.path.join(output_dir, f"{name_without_ext}_plot.png")
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close()

        # ==========================================
        # 2. ENERGY PLOT (CALIBRATED)
        # ==========================================
        plt.figure(figsize=(10, 5))
        plt.plot(energy, counts, drawstyle='steps-mid', color='blue', linewidth=1)
        plt.xlim(np.min(energy), np.max(energy))
        plt.yscale('log')
        plt.ylim(min_val, max_val * 1.2)
        plt.xlabel(f'Energy ({energy_unit})')
        plt.ylabel('Counts (Log Scale)')
        plt.title(
            f'Calibrated Spectrum: {base_name}\n(Energy Range: {np.min(energy):.1f} - {np.max(energy):.1f} {energy_unit})')
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()

        cal_output_filename = os.path.join(cal_output_dir, f"{name_without_ext}_cal_plot.png")
        plt.savefig(cal_output_filename, dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"\nFailed to process {filepath}: {e}")

print(f"\nAll plots saved successfully:")
print(f" - Uncalibrated plots: '{output_dir}'")
print(f" - Calibrated plots: '{cal_output_dir}'")