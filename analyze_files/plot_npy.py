import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm


def unnormalize_spectrum(spectrum, y_vector, live_time_seconds=3600):
    """
    Converts MCNP fractional probabilities to expected physical counts.
    Assumes y_vector contains activities in Becquerels (decays/second).
    Default live time is set to 3600 seconds (1 hour).
    """
    # Calculate the total activity of all isotopes in the sample
    total_activity_bq = np.sum(y_vector)

    # Expected counts = probability_per_decay * total_decays
    physical_counts = spectrum * total_activity_bq * live_time_seconds
    return physical_counts


def plot_spectrum(spectrum, sample_idx, save_dir):
    """Visualizes the 8192-channel HPGe spectrum on a log scale."""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Clip data to a minimum of 1 so empty channels plot safely on a log scale
    spectrum_clipped = np.clip(spectrum, a_min=1, a_max=None)

    ax.plot(spectrum_clipped, color='blue', linewidth=0.8, alpha=0.8)
    ax.set_title(f"MCNP Simulated HPGe Spectrum | Sample Index: {sample_idx:03d}")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts (Log Scale)")

    # Apply log scale and force the bottom of the axis to exactly 1 (10^0)
    ax.set_yscale('log')
    ax.set_ylim(bottom=1)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    # Formats the filename to zero-padded 3-digit number (e.g., 001_plot.png)
    save_path = os.path.join(save_dir, f"{sample_idx:03d}_plot.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_y_labels(y_vector, sample_idx, save_dir, isotope_names):
    """Visualizes the 41-dimensional activity label vector with isotope names."""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Bar chart for the 41 activity levels mapped to their real isotope names
    ax.bar(isotope_names, y_vector, color='coral', edgecolor='black')

    ax.set_title(f"Radionuclide Activity Levels | Sample Index: {sample_idx:03d}")
    ax.set_xlabel("Radionuclides (41 Isotopes)")
    ax.set_ylabel("Activity Level")

    # Rotate the x-labels so all 41 names fit clearly without overlapping
    plt.xticks(rotation=90, fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    # Formats the filename to zero-padded 3-digit number (e.g., 001_activity.png)
    save_path = os.path.join(save_dir, f"{sample_idx:03d}_activity.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    # 1. Setup Paths
    base_dir = "data/MCNP_HPGe"
    data_dir = os.path.join(base_dir, "npy")

    # Define the output directories OUTSIDE of npy_csv
    plots_dir = os.path.join(base_dir, "plots")
    activities_dir = os.path.join(base_dir, "activities")

    # Define paths for loading the data INSIDE npy_csv
    matrix_path = os.path.join(data_dir, "x.npy")
    y_labels_path = os.path.join(data_dir, "y.npy")  # Assuming y is saved here
    activity_csv_path = os.path.join(base_dir, "Activity.csv")  # Used for isotope names

    # Create both directories safely
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(activities_dir, exist_ok=True)

    # 2. Load the Spectrum Matrix (X)
    try:
        spectrum_matrix = np.load(matrix_path)
    except FileNotFoundError:
        print(f"Error: Could not find {matrix_path}.")
        return

    # 3. Load the Activity Labels (y)
    try:
        y_matrix = np.load(y_labels_path)
    except FileNotFoundError:
        print(f"Error: Could not find {y_labels_path}.")
        return

    # 4. Extract Isotope Names from Activity.csv
    try:
        df = pd.read_csv(activity_csv_path)
        isotope_names = df.columns.tolist()
        if len(isotope_names) != 41:
            print("Warning: Activity.csv does not have exactly 41 columns. Check your file format.")
            # Fallback to generic names if the CSV structure is unexpected
            isotope_names = [f"Iso-{i + 1}" for i in range(len(y_matrix[0]))]
    except FileNotFoundError:
        print(f"Warning: Could not find {activity_csv_path}. Using generic labels for isotopes.")
        isotope_names = [f"Iso-{i + 1}" for i in range(41)]

    # 5. Select number of plots to create
    num_samples = len(spectrum_matrix)

    print(f"Starting to plot {num_samples} samples.")
    print(f" - Saving spectra to: {plots_dir}/")
    print(f" - Saving activities to: {activities_dir}/")

    # Loop over all samples and route the plots to their respective folders
    for sample_idx in tqdm(range(num_samples), desc="Generating Plots"):
        # Unnormalize the spectrum before plotting
        unnormalized_spectrum = unnormalize_spectrum(
            spectrum_matrix[sample_idx],
            y_matrix[sample_idx],
            live_time_seconds=3600  # Adjust this if your assumed collection time is different
        )

        plot_spectrum(unnormalized_spectrum, sample_idx, plots_dir)
        plot_y_labels(y_matrix[sample_idx], sample_idx, activities_dir, isotope_names)

    print("\nSuccessfully finished saving all plots and activities!")


if __name__ == "__main__":
    main()