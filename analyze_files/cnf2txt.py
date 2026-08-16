import os
import glob
from tqdm import tqdm
from read_cnf import read_cnf_file

# ==========================================
# SETUP DIRECTORIES
# ==========================================
dataset_dir = "data/Baq_HPGe/cnf"
output_dir = "data/Baq_HPGe/txt"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all CNF files (case-insensitive) and sort them
all_cnf_files = glob.glob(os.path.join(dataset_dir, "*.[cC][nN][fF]"))
all_cnf_files.sort()

print(f"Found {len(all_cnf_files)} CNF files. Converting to TXT...")

# ==========================================
# CONVERSION LOOP
# ==========================================
for filepath in tqdm(all_cnf_files, desc="Converting CNF to TXT"):
    try:
        # Parse the CNF file using your custom reader
        cnf_data = read_cnf_file(filepath)
        counts = cnf_data.get('Channels data')

        if counts is None or len(counts) == 0:
            print(f"\nWarning: Could not find 'Channels data' in {filepath}. Skipping.")
            continue

        # Prepare the output text filename
        base_name = os.path.basename(filepath)
        name_without_ext = os.path.splitext(base_name)[0]
        output_filename = os.path.join(output_dir, f"{name_without_ext}.txt")

        # Write the counts to a text file in the required format
        with open(output_filename, 'w') as txt_file:
            # 1. Write the header exactly as requested (using a tab separator)
            txt_file.write("Channel\tCounts\n")

            # 2. Iterate using enumerate to get both the index and the count
            for channel_idx, count in enumerate(counts):
                txt_file.write(f"{channel_idx}\t{int(count)}\n")

    except Exception as e:
        print(f"\nFailed to process {filepath}: {e}")

print(f"\nAll files successfully converted and saved to '{output_dir}'.")