import sys
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from read_cnf import read_cnf_file

# --- File Paths ---
cnf_path = 'data/Baq_HPGe/fc17344-160724-AEGIS.CNF'
rpt_path = 'data/Baq_HPGe/fc17344-160724-AEGIS.RPT'


# --- Parsing Functions ---
def parse_rpt_metadata(filepath):
    """Extracts key metadata from the RPT file."""
    params = {"Sample Title": "Unknown", "Detector": "Unknown",
              "Acq Started": "Unknown", "Live Time": "Unknown", "Dead Time": "Unknown"}
    try:
        with open(filepath, 'r') as file:
            content = file.read()
            if m := re.search(r"Sample Title\s*:\s*(.+)", content): params["Sample Title"] = m.group(1).strip()
            if m := re.search(r"Detector Name:\s*(.+)", content): params["Detector"] = m.group(1).strip()
            if m := re.search(r"Acquisition Started\s*:\s*(.+)", content): params["Acq Started"] = m.group(1).strip()
            if m := re.search(r"Live Time\s*:\s*(.+)", content): params["Live Time"] = m.group(1).strip()
            if m := re.search(r"Dead Time\s*:\s*(.+)", content): params["Dead Time"] = m.group(1).strip()
    except Exception as e:
        print(f"Metadata read error: {e}")
    return params


def parse_rpt_isotopes(filepath):
    """Extracts isotopes and activity from the Interference Corrected table."""
    isotopes = {}
    try:
        with open(filepath, 'r') as file:
            content = file.read()
            # Split to get only the final corrected table
            table_split = content.split("I N T E R F E R E N C E   C O R R E C T E D   R E P O R T")
            if len(table_split) > 1:
                target_text = table_split[1]
                # Regex to grab Name and Activity (Bq/Unit)
                pattern = re.compile(
                    r"^\s+([A-Za-z]{1,2}-\d{2,3}[m]?)\s+(?:[@X\?]\s+)?(\d\.\d+)\s+(\d\.\d+E[+-]\d{2,3})", re.MULTILINE)
                for match in pattern.finditer(target_text):
                    isotopes[match.group(1).strip()] = float(match.group(3))
    except Exception as e:
        print(f"Isotope read error: {e}")
    # Sort by activity descending for the legend
    return dict(sorted(isotopes.items(), key=lambda item: item[1], reverse=True))


# --- Main Plotting Routine ---
try:
    # 1. Load Data
    print("Processing files...")
    cnf_data = read_cnf_file(cnf_path)
    counts = cnf_data.get('Channels data')

    metadata = parse_rpt_metadata(rpt_path)
    isotopes = parse_rpt_isotopes(rpt_path)

    if counts is not None:
        # 2. Setup Plot
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.plot(counts, drawstyle='steps-mid', color='#004488', linewidth=0.8)

        ax.set_yscale('log')
        ax.set_xlabel('Channel Number', fontweight='bold')
        ax.set_ylabel('Counts', fontweight='bold')
        ax.set_title('AEGIS Gamma-Ray Spectrum Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, which="both", linestyle="--", alpha=0.6)

        # 3. Create Legend 1: Metadata (Upper Right)
        meta_text = (
            f"Sample: {metadata['Sample Title']}\n"
            f"Detector: {metadata['Detector']}\n"
            f"Acquired: {metadata['Acq Started']}\n"
            f"Live Time: {metadata['Live Time']}\n"
            f"Dead Time: {metadata['Dead Time']}"
        )
        # We attach the string to an invisible line to force it into a legend
        fake_line = Line2D([0], [0], color='none', label=meta_text)
        leg1 = ax.legend(handles=[fake_line], loc='upper right', framealpha=0.9,
                         edgecolor='black', prop={'family': 'monospace', 'size': 10})

        # MUST ADD ARTIST so the second legend doesn't overwrite it
        ax.add_artist(leg1)

        # 4. Create Legend 2: Isotopes (Upper Left)
        if isotopes:
            isotope_handles = []
            for iso, activity in isotopes.items():
                # Format to 2 decimal places (e.g., "1387.00 Bq")
                label_str = f"{iso:<8} {activity:>8.2f} Bq"
                isotope_handles.append(Line2D([0], [0], color='none', label=label_str))

            ax.legend(handles=isotope_handles, loc='upper left', title="Identified Isotopes (Activity)",
                      title_fontproperties={'weight': 'bold'}, framealpha=0.9, edgecolor='black',
                      prop={'family': 'monospace', 'size': 10})

        plt.tight_layout()
        plt.show()

        # Force cleanup so the program ends cleanly when you close the window
        plt.close('all')
        sys.exit(0)

    else:
        print("Error: Could not find 'Channels data' in the CNF file.")

except Exception as e:
    print(f"Failed to process spectrum: {e}")