import numpy as np
from pathlib import Path

# Base directories containing both the spectra and coeff folders
data_dirs = [
    Path("data/IAEA/2024"),
    Path("data/IAEA/2023"),
    Path("data/Baq")

]

out_spectra = Path("data/IAEA_Baq/spectrum_matrix.npy")
out_coeffs = Path("data/IAEA_Baq/coeff_matrix.npy")

all_counts = []
all_coeffs = []
L = 0  # Maximum channel length

for base_dir in data_dirs:
    # IMPORTANT: If you renamed the 'txt' folder to 'spectra', update the string below
    spectra_dir = base_dir / "spectra_txt"
    coeff_dir = base_dir / "coeff_txt"

    if not spectra_dir.exists():
        print(f"Warning: Directory {spectra_dir} not found. Skipping.")
        continue

    # Sort files to ensure deterministic alignment across runs
    for txt_file in sorted(spectra_dir.glob("*.txt")):
        coeff_file = coeff_dir / txt_file.name

        if not coeff_file.exists():
            print(f"Warning: Missing coefficient file for {txt_file.name}. Skipping.")
            continue

        # 1. Parse Spectrum
        counts = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.strip()
                if clean_line.startswith('#') or clean_line.startswith('Channel'):
                    continue
                parts = clean_line.split()
                if len(parts) >= 2:
                    counts.append(int(parts[1]))

        all_counts.append(counts)
        if len(counts) > L:
            L = len(counts)

        # 2. Parse Calibration Coefficients
        coeffs = np.zeros(4)
        with open(coeff_file, 'r', encoding='utf-8') as f:
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

        all_coeffs.append(coeffs)

N = len(all_counts)

if N == 0:
    print("No valid file pairs found in any of the provided directories.")
else:
    # 3. Create the arrays (int32 for counts, float32 for parameters)
    matrix_spectra = np.zeros((N, L), dtype=np.int32)
    matrix_coeffs = np.zeros((N, 4), dtype=np.float32)

    # 4. Populate the matrices
    for i, counts in enumerate(all_counts):
        matrix_spectra[i, :len(counts)] = counts
        matrix_coeffs[i, :] = all_coeffs[i]

    # 5. Save everything
    out_spectra.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_spectra, matrix_spectra)
    np.save(out_coeffs, matrix_coeffs)

    print(f"Successfully processed {N} paired files.")
    print(f"Spectra Matrix saved to {out_spectra} | Shape: {matrix_spectra.shape}")
    print(f"Coeffs Matrix saved to {out_coeffs}   | Shape: {matrix_coeffs.shape}")