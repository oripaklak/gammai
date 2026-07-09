import os
os.environ['NPY_PROMOTION_STATE'] = 'legacy'

import matplotlib.pyplot as plt
from read_cnf import read_cnf_file
# ... rest of your code ...
import matplotlib.pyplot as plt
from read_cnf import read_cnf_file

file_path = 'data/CTBT_HPGe/fc17344-160724-AEGIS.CNF'

try:
    # 1. Bypass becquerel and read the binary file directly
    print(f"Reading {file_path}...")
    cnf_data = read_cnf_file(file_path)

    # 2. Extract the raw spectrum counts
    counts = cnf_data.get('Channels data')

    if counts is not None:
        # 3. Plot the spectrum
        plt.figure(figsize=(12, 6))

        # 'steps-mid' is the standard way to visualize histogram bin data in spectroscopy
        plt.plot(counts, drawstyle='steps-mid', color='#004488', linewidth=0.8)

        # Set a logarithmic scale for the Y-axis
        plt.yscale('log')

        plt.xlabel('Channel Number')
        plt.ylabel('Counts')
        plt.title('AEGIS Gamma-Ray Spectrum (32k Channels)')

        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
    else:
        print("Error: Could not find 'Channels data' in the parsed CNF file.")

except Exception as e:
    print(f"Failed to process spectrum: {e}")