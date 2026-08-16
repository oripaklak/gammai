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
output_dir = "data/Baq_HPGe/plots1"

os.makedirs(output_dir, exist_ok=True)

all_cnf_files = glob.glob(os.path.join(dataset_dir, "*.[cC][nN][fF]"))
all_cnf_files.sort()

print(f"Found {len(all_cnf_files)} CNF files. Generating selectively truncated plots...")

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

        # --- Truncate the signal where a sequence of zeros begins ---
        window_size = 50  # Number of consecutive zeros that defines a "sequence"
        is_zero = (counts == 0).astype(int)

        # Calculate a rolling sum to find where a window of all zeros exists
        zero_runs = np.convolve(is_zero, np.ones(window_size, dtype=int), mode='valid')
        seq_indices = np.where(zero_runs == window_size)[0]

        if len(seq_indices) > 0:
            # Truncate the array exactly at the first zero of the sequence
            truncate_idx = seq_indices[0]
            counts = counts[:truncate_idx]
        else:
            # Fallback: If there is no sequence of 50 zeros, truncate trailing zeros at the end
            non_zero_indices = np.nonzero(counts)[0]
            if len(non_zero_indices) > 0:
                counts = counts[:non_zero_indices[-1] + 1]

        plt.figure(figsize=(10, 5))

        # Plot the truncated array
        plt.plot(counts, drawstyle='steps-mid', color='blue', linewidth=1)

        # Set X-axis to exactly match the length of the newly truncated array
        plt.xlim(0, len(counts))
        plt.yscale('log')

        # Calculate min and max values dynamically for the Y-axis based on truncated data
        min_val = max(1, np.min(counts))
        max_val = np.max(counts)
        plt.ylim(min_val, max_val * 1.2)

        plt.xlabel('Channel Number')
        plt.ylabel('Counts (Log Scale)')

        # Add original length to the title
        base_name = os.path.basename(filepath)
        plt.title(f'Spectrum: {base_name}\n(Original Length: {original_length} Channels)')
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()

        # Save output file
        name_without_ext = os.path.splitext(base_name)[0]
        output_filename = os.path.join(output_dir, f"{name_without_ext}_plot.png")

        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"\nFailed to process {filepath}: {e}")

print(f"\nAll CNF plots saved successfully to '{output_dir}'.")