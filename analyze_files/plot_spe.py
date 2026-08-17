import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def get_file_number(filepath):
    """Extracts the integer number from the filename to ensure strict '1, 2, 3...' sorting."""
    filename = os.path.basename(filepath)
    try:
        return int(filename.replace('.spe', ''))
    except ValueError:
        return -1


def read_spe_file(filepath):
    """Parses an .spe file and returns a list of counts and calibration parameters."""
    counts = []
    ener_fit = (0.0, 1.0)  # Default: b=0, a=1 (1:1 channel mapping) if missing

    in_data = False
    in_ener_fit = False

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()

            # Extract ENER_FIT parameters
            if in_ener_fit:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        b = float(parts[0])
                        a = float(parts[1])
                        ener_fit = (b, a)
                    except ValueError:
                        pass
                in_ener_fit = False  # Only read the first line after the tag

            if line == '$ENER_FIT:':
                in_ener_fit = True

            # Extract DATA
            if in_data and line.startswith('$'):
                in_data = False

            if in_data:
                # Skip the length definition line (e.g., "0 8191")
                if len(line.split()) > 1:
                    continue
                try:
                    counts.append(int(line))
                except ValueError:
                    pass

            if line == '$DATA:':
                in_data = True

    return counts, ener_fit


# ==========================================
# SETUP AND DATA LOADING
# ==========================================

# --- NEW PARAMETER ---
calibration = True

dataset_dir = "data/IAEA_HPGe/2024/spe"
# Change output directory based on the calibration flag
output_dir = "data/IAEA_HPGe/2024/cal_plots" if calibration else "data/IAEA_HPGe/2024/plots"

os.makedirs(output_dir, exist_ok=True)

all_spe_files = glob.glob(os.path.join(dataset_dir, "*.spe"))
all_spe_files.sort(key=get_file_number)

print(f"Found {len(all_spe_files)} files. Generating individual plots in order...")

# ==========================================
# PLOTTING LOOP
# ==========================================

for filepath in tqdm(all_spe_files, desc="Saving Plots"):
    counts, (b, a) = read_spe_file(filepath)
    file_num = get_file_number(filepath)

    if not counts:
        continue

    # --- CALIBRATION LOGIC ---
    num_channels = len(counts)
    if calibration:
        # x_2 = a * x_1 + b
        channels = np.arange(num_channels)
        x_axis = a * channels + b
        x_label = 'Energy'
    else:
        x_axis = np.arange(num_channels)
        x_label = 'Channel Number'

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, counts, color='blue', linewidth=1)

    plt.yscale('log')

    min_val = max(1, min(counts))
    max_val = max(counts)

    plt.ylim(min_val, max_val * 1.2)
    plt.xlim(x_axis[0], x_axis[-1])

    plt.xlabel(x_label)
    plt.ylabel('Counts (Log Scale)')
    title_suffix = "(Calibrated)" if calibration else "(Uncalibrated)"
    plt.title(f'Spectrum Sequence: File {file_num}.spe {title_suffix}')
    plt.grid(True, which="both", ls="--", alpha=0.5)

    output_filename = os.path.join(output_dir, f"{file_num:04d}_plot.png")

    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()

print(f"\nAll plots saved successfully to '{output_dir}'.")