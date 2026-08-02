import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_file_number(filepath):
    """Extracts the integer number from the filename to ensure strict '1, 2, 3...' sorting."""
    filename = os.path.basename(filepath)
    try:
        # Strips out '.spe' and converts to an integer
        return int(filename.replace('.spe', ''))
    except ValueError:
        return -1


def read_spe_file(filepath):
    """Parses an .spe file and returns a list of counts."""
    counts = []
    in_data = False
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if in_data and line.startswith('$'):
                break
            if in_data:
                if len(line.split()) > 1:
                    continue
                try:
                    counts.append(int(line))
                except ValueError:
                    pass
            if line == '$DATA:':
                in_data = True
    return counts


# ==========================================
# SETUP AND DATA LOADING
# ==========================================

dataset_dir = "data/IAEA_HPGe/2024/spe"
output_dir = "data/IAEA_HPGe/2024/plots"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Find all files and sort them NUMERICALLY
all_spe_files = glob.glob(os.path.join(dataset_dir, "*.spe"))
all_spe_files.sort(key=get_file_number)

print(f"Found {len(all_spe_files)} files. Generating individual plots in order...")

# ==========================================
# PLOTTING LOOP
# ==========================================

# Using tqdm to show a progress bar so you know how long it will take
for filepath in tqdm(all_spe_files, desc="Saving Plots"):
    counts = read_spe_file(filepath)
    file_num = get_file_number(filepath)

    # Skip empty files to prevent plotting errors
    if not counts:
        continue

    plt.figure(figsize=(10, 5))
    plt.plot(counts, color='blue', linewidth=1)

    plt.yscale('log')

    # Calculate min and max values dynamically for the Y-axis
    # max(1, ...) ensures we never pass 0 or a negative number to the log scale
    min_val = max(1, min(counts))
    max_val = max(counts)

    # Set Y-limits from the exact min to max (with a 20% buffer on top for visibility)
    plt.ylim(min_val, max_val * 1.2)

    # Optionally, lock the X-axis from 0 to the maximum channel number
    plt.xlim(0, len(counts) - 1)

    plt.xlabel('Channel Number')
    plt.ylabel('Counts (Log Scale)')
    plt.title(f'Spectrum Sequence: File {file_num}.spe')
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # Save the file with zero-padding (e.g., 0001_plot.png, 0002_plot.png)
    output_filename = os.path.join(output_dir, f"{file_num:04d}_plot.png")

    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()  # CRITICAL: Clears memory so your computer doesn't crash

print(f"\nAll plots saved successfully to '{output_dir}'.")