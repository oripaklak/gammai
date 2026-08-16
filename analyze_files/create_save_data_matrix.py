import numpy as np
from pathlib import Path

data_dirs = [
    Path("data/IAEA_HPGe/2024/txt"),
    Path("data/IAEA_HPGe/2023/txt"),
    Path("data/Baq_HPGe/txt")
]
output_file = Path("data/spectrum_matrix_IAEA_Baq.npy")

# Get a list of all text files from ALL directories
txt_files = []
for directory in data_dirs:
    if directory.exists():
        txt_files.extend(list(directory.glob("*.txt")))
    else:
        print(f"Warning: Directory {directory} not found.")

N = len(txt_files)

if N == 0:
    print("No .txt files found in any of the provided directories.")
else:
    all_counts = []
    L = 0  # To track the maximum channel length

    # 1. Read all files and extract just the counts
    for txt_file in txt_files:
        counts = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.strip()

                # Skip comments and header lines
                if clean_line.startswith('#') or clean_line.startswith('Channel'):
                    continue

                # Split by tab/space and grab the second column (Counts)
                parts = clean_line.split()
                if len(parts) >= 2:
                    counts.append(int(parts[1]))

        all_counts.append(counts)

        # Update maximum length L if this file is longer
        if len(counts) > L:
            L = len(counts)

    # 2. Create an N x L matrix filled with zeros
    # Using int32 since HPGe spectrum counts are integers
    matrix = np.zeros((N, L), dtype=np.int32)

    # 3. Populate the matrix with the data
    for i, counts in enumerate(all_counts):
        # Insert the counts into the ith row, up to the length of that specific file
        matrix[i, :len(counts)] = counts

    # 4. Save the matrix to an .npy file
    output_file.parent.mkdir(parents=True, exist_ok=True) # Ensure the output folder exists
    np.save(output_file, matrix)

    print(f"Matrix successfully created and saved to {output_file}")
    print(f"Shape of the combined matrix: N = {N} (examples), L = {L} (max channels)")