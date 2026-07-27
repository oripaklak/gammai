import os
from pathlib import Path

# Define directories
source_dir = Path("data/IAEA_HPGe/spe")
target_dir = Path("data/IAEA_HPGe/txt")

# Create the target directory
target_dir.mkdir(parents=True, exist_ok=True)

# Process each .spe file
processed_count = 0
for spe_file in source_dir.glob("*.spe"):
    txt_file = target_dir / spe_file.with_suffix(".txt").name

    # Read the original .spe file
    with open(spe_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    in_data_block = False
    channel_data = []
    start_channel = 0

    for i, line in enumerate(lines):
        clean_line = line.strip()

        # Look for the start of the data block
        if clean_line.startswith("$DATA:"):
            in_data_block = True
            # The next line is the channel range (e.g., "0 8191")
            try:
                dims = lines[i + 1].strip().split()
                start_channel = int(dims[0])
            except (IndexError, ValueError):
                start_channel = 0
            continue

        if in_data_block:
            # Skip the dimension line itself since we just parsed it
            if i > 0 and lines[i - 1].strip().startswith("$DATA:"):
                continue

            # If we hit the next header block (like $ENER_FIT:), stop reading
            if clean_line.startswith("$"):
                break

            # Parse the counts
            if clean_line:
                channel_data.extend([int(x) for x in clean_line.split() if x.lstrip('-').isdigit()])

    # Write the extracted data to the new .txt file
    with open(txt_file, 'w') as f:
        f.write("Channel\tCounts\n")  # Header for your text file
        for j, count in enumerate(channel_data):
            f.write(f"{start_channel + j}\t{count}\n")

    processed_count += 1

print(f"Successfully extracted data for {processed_count} files into {target_dir}")