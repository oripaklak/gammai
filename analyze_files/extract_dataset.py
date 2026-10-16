import os
import glob
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from read_cnf import read_cnf_file  # Ensure this is in the same directory or Python path


def write_counts(output_path, counts, start_channel=0):
    """Helper to write spectrum counts."""
    with open(output_path, 'w') as f:
        f.write("Channel\tCounts\n")
        for j, count in enumerate(counts):
            f.write(f"{start_channel + j}\t{int(count)}\n")


def write_coeffs(output_path, coeffs):
    """Helper to write 4 calibration parameters."""
    with open(output_path, 'w') as f:
        for i in range(4):
            f.write(f"c{i}: {coeffs[i]}\n")


def process_spe(filepath, txt_path, coeff_path):
    """Parses a single .spe file for data and ENER_FIT."""
    counts = []
    c0, c1 = 0.0, 1.0  # Default if missing
    start_channel = 0
    in_data, in_ener = False, False

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        clean_line = line.strip()

        # Extract Calibration
        if in_ener:
            parts = clean_line.split()
            if len(parts) >= 2:
                try:
                    c0 = float(parts[0])  # b
                    c1 = float(parts[1])  # a
                except ValueError:
                    pass
            in_ener = False
        elif clean_line == '$ENER_FIT:':
            in_ener = True

        # Extract Data
        if clean_line.startswith("$DATA:"):
            in_data = True
            try:
                start_channel = int(lines[i + 1].strip().split()[0])
            except (IndexError, ValueError):
                pass
            continue

        if in_data:
            if i > 0 and lines[i - 1].strip().startswith("$DATA:"):
                continue  # Skip dimension line
            if clean_line.startswith("$"):
                in_data = False  # Next block started
                continue
            if clean_line:
                counts.extend([int(x) for x in clean_line.split() if x.lstrip('-').isdigit()])

    coeffs = [c0, c1, 0.0, 0.0]

    # Write Outputs
    write_counts(txt_path, counts, start_channel)
    write_coeffs(coeff_path, coeffs)

    return coeffs


def process_cnf(filepath, txt_path, coeff_path):
    """Parses a single .cnf file for data and Energy coefficients."""
    cnf_data = read_cnf_file(filepath)
    counts = cnf_data.get('Channels data', [])

    if len(counts) == 0:
        raise ValueError("No 'Channels data' found.")

    # Extract and pad coefficients to exactly 4 items
    raw_coeffs = cnf_data.get('Energy coefficients', [])
    coeffs = np.zeros(4)
    length = min(len(raw_coeffs), 4)
    if length > 0:
        coeffs[:length] = raw_coeffs[:length]

    # Write Outputs
    write_counts(txt_path, counts, start_channel=0)
    write_coeffs(coeff_path, coeffs)

    return coeffs.tolist()


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Define all target directories in a list
    data_dirs = [
        "results/diff_9M_L36_C128_bs16_cosine-0.01_ep1000_calib_film_Baq/weights-80000_2026-10-05_11-10-58/cnf",
    ]

    global_coeffs = []  # Master list to track everything

    for data_dir in data_dirs:
        print(f"\n{'=' * 50}\nProcessing Directory: {data_dir}\n{'=' * 50}")
        source_path = Path(data_dir)

        if not source_path.exists():
            print(f"Warning: Directory {data_dir} does not exist. Skipping.")
            continue

        # Auto-create output directories in the parent folder of each dataset
        parent_dir = source_path.parent
        txt_out = parent_dir / "spectra_txt"
        coeff_out = parent_dir / "coeff_txt"

        txt_out.mkdir(parents=True, exist_ok=True)
        coeff_out.mkdir(parents=True, exist_ok=True)

        # Detect file types
        spe_files = list(source_path.glob("*.spe"))
        cnf_files = list(source_path.glob("*.[cC][nN][fF]"))

        if spe_files:
            files = spe_files
            processor = process_spe
            print(f"Detected {len(files)} SPE files.")
        elif cnf_files:
            files = cnf_files
            processor = process_cnf
            print(f"Detected {len(files)} CNF files.")
        else:
            print(f"No .spe or .cnf files found in {data_dir}. Skipping.")
            continue

        # Run Extraction Loop
        success_count = 0
        for filepath in tqdm(files, desc=f"Extracting {source_path.name}"):
            filename = filepath.stem
            txt_target = txt_out / f"{filename}.txt"
            coeff_target = coeff_out / f"{filename}.txt"

            try:
                # Process the file and capture the returned coefficients
                file_coeffs = processor(str(filepath), str(txt_target), str(coeff_target))
                global_coeffs.append(file_coeffs)
                success_count += 1
            except Exception as e:
                print(f"\nFailed to process {filename}: {e}")

        print(f"Successfully extracted {success_count} files.")
        print(f"Spectra saved to: {txt_out}")
        print(f"Coefficients saved to: {coeff_out}")

    # ==========================================
    # CALCULATE AND SAVE GLOBAL BOUNDS
    # ==========================================
    if global_coeffs:
        print(f"\n{'=' * 50}\nCalculating Global Calibration Bounds\n{'=' * 50}")
        coeff_matrix = np.array(global_coeffs)

        calib_min = np.min(coeff_matrix, axis=0).tolist()
        calib_max = np.max(coeff_matrix, axis=0).tolist()

        bounds_dict = {
            "calib_min": calib_min,
            "calib_max": calib_max
        }

        bounds_file = "data/IAEA_Baq/global_calib_bounds.json"

        # Ensure the data directory exists just in case
        os.makedirs("data", exist_ok=True)

        with open(bounds_file, "w") as f:
            json.dump(bounds_dict, f, indent=4)

        print(f"Global bounds successfully saved to: {bounds_file}")
        print(f"Min: {calib_min}")
        print(f"Max: {calib_max}")