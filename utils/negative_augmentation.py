import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def offset_spectrum(spectrum, x_shift=0, y_shift=0.0):
    """
    Applies an offset to the x-axis (channel shift) and/or y-axis (count shift).
    """
    aug_spectrum = np.copy(spectrum).astype(np.float32)

    # Apply Y-Axis Offset
    if y_shift != 0.0:
        aug_spectrum += y_shift
        # Clip to 0 since physical counts cannot be negative
        aug_spectrum = np.clip(aug_spectrum, a_min=0.0, a_max=None)

    # Apply X-Axis Offset
    if x_shift != 0:
        shifted = np.zeros_like(aug_spectrum)
        if x_shift > 0:
            shifted[x_shift:] = aug_spectrum[:-x_shift]
        else:
            shifted[:x_shift] = aug_spectrum[-x_shift:]
        aug_spectrum = shifted

    return aug_spectrum


def flip_spectrum(spectrum):
    """
    Flips the spectrum horizontally around its center point (x = L/2).
    This reverses the array, placing the low-energy continuum at the high-energy end.
    """
    return spectrum[::-1].copy()


def truncate_spectrum(spectrum, drop_fraction=0.15):
    """
    Crops the signal and pads it back to the original length with zeros.
    """
    original_length = len(spectrum)
    new_length = int(original_length * (1 - drop_fraction))

    truncated = np.zeros_like(spectrum)
    truncated[new_length:] = spectrum[new_length:]

    return truncated


def crop_spectrum(spectrum, window_size=500, start_idx=None):
    """
    Slides a window over the x-axis and blanks out (zeros) that section of the signal.
    """
    aug_spectrum = np.copy(spectrum)
    length = len(aug_spectrum)

    if start_idx is None:
        start_idx = np.random.randint(0, length - window_size)

    aug_spectrum[start_idx: start_idx + window_size] = 0.0

    return aug_spectrum


def load_random_real_spectrum(data_dir="data/Baq_HPGe/txt"):
    """
    Scans the provided directory and loads a random real .txt spectrum.
    """
    folder_path = Path(data_dir)
    if not folder_path.exists():
        raise FileNotFoundError(f"Directory {data_dir} not found. Please run from the project root.")

    txt_files = list(folder_path.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {data_dir}")

    random_file = random.choice(txt_files)
    print(f"Selected real spectrum: {random_file}")

    # Load the data, skipping the "Channel" header
    spectrum = np.loadtxt(random_file, dtype=np.float32, skiprows=1, usecols=1)

    return spectrum, random_file.name


def main():
    """
    Picks a random real signal, applies all negative augmentations,
    and plots them in a 2x2 grid, with the original in a separate window.
    """
    # 1. Load the original real signal
    try:
        original, file_name = load_random_real_spectrum()
    except Exception as e:
        print(f"Error: {e}")
        return

    original_L = len(original)

    # 2. Apply augmentations
    offset_sig = offset_spectrum(original, x_shift=300, y_shift=2000)
    flipped_sig = flip_spectrum(original)
    truncated_sig = truncate_spectrum(original, drop_fraction=0.25)
    cropped_sig = crop_spectrum(original, window_size=600)

    # Find a good Y-axis maximum based on the real data
    ymax = max(np.max(original) * 2.0, 1000.0)
    ymin = 1.0

    # Common plot settings helper
    def format_ax(ax, title, sig, color='blue', is_truncated=False):
        ax.plot(sig, color=color, linewidth=1)
        ax.set_yscale('log')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.set_ylabel("Counts (Log)")

        # Keep X-limits consistent unless it is the truncated signal
        if is_truncated:
            ax.set_xlim(0, len(sig))
            ax.axvline(len(sig), color='red', linestyle='--', label='Truncated Edge')
            ax.legend(loc='upper right')
        else:
            ax.set_xlim(0, original_L)

        ax.set_ylim(bottom=ymin, top=ymax)

    # 3. Create Window 1: Original Spectrum
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    fig1.canvas.manager.set_window_title('Original Spectrum')
    format_ax(ax1, f"Original Spectrum ({file_name}, Length: {original_L})", original, color='black')
    fig1.tight_layout()

    # 4. Create Window 2: 2x2 Augmentations Grid
    fig2, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig2.canvas.manager.set_window_title('Negative Augmentations (NDA)')

    # Top Left: Offset
    format_ax(axes[0, 0], "Offset (x_shift=+300, y_shift=+2000)", offset_sig, color='purple')

    # Top Right: Flipped
    format_ax(axes[0, 1], f"Flipped (around x = {original_L // 2})", flipped_sig, color='orange')

    # Bottom Left: Truncated
    format_ax(axes[1, 0], f"Truncated (Length ? 2^x: {len(truncated_sig)} channels)", truncated_sig, color='red',
              is_truncated=True)

    # Bottom Right: Cropped
    format_ax(axes[1, 1], "Cropped (Blanking Window)", cropped_sig, color='green')

    fig2.tight_layout()

    # Show both windows simultaneously
    plt.show()


if __name__ == "__main__":
    main()