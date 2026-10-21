import json
import os
import signal
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_noise_schedule(beta, alpha_bar, run_dir):
    """Generates and saves visual plots for the diffusion noise schedules."""
    save_dir = os.path.join(run_dir, "tracked_noise")
    os.makedirs(save_dir, exist_ok=True)

    T = len(beta)
    steps = np.arange(T)

    plt.figure(figsize=(10, 5))
    plt.plot(
        steps, beta, label="Beta (Variance added)", color="red", linewidth=2
    )
    plt.title(f"Beta Noise Schedule (T={T})")
    plt.xlabel("Diffusion Step (t)")
    plt.ylabel("Beta Value")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "beta_schedule.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        steps,
        alpha_bar,
        label="Alpha Bar (Signal Preserved)",
        color="blue",
        linewidth=2,
    )
    plt.title(f"Cumulative Signal Preservation (T={T})")
    plt.xlabel("Diffusion Step (t)")
    plt.ylabel("Alpha Bar Value")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "alpha_cum_schedule.png"), dpi=300)
    plt.close()

    print(f"Noise schedule plots saved to: {save_dir}/")

def plot_and_save_spectra(
    generated_counts,
    save_dir,
    model_name,
    gen_coeffs=None,
    detect_peaks=False,
    min_prominence=25.0,
    bounds_path=None,
):
    """Plots final generated spectra in Energy (keV) space."""
    os.makedirs(save_dir, exist_ok=True)
    num_samples = len(generated_counts)

    for i in range(num_samples):
        spectrum = generated_counts[i]
        coeffs = gen_coeffs[i] if gen_coeffs is not None else None

        L = len(spectrum)
        channels = np.arange(L)

        fig, ax = plt.subplots(figsize=(12, 4.5))

        if coeffs is not None and len(coeffs) == 4:
            x_axis = compute_safe_energies(coeffs, L)
            x_label = "Energy (keV)"
            max_e = np.max(x_axis) if len(x_axis) > 0 else 3000.0
            xlim_right = float(np.clip(max_e, 30.0, 3000.0))
            ax.set_xlim(0.0, xlim_right)
        else:
            x_axis = channels
            x_label = "Channel"
            ax.set_xlim(0, L)

        sig_plot = np.maximum(spectrum, 1e-1)
        ax.plot(
            x_axis,
            sig_plot,
            color="blue",
            linewidth=1,
            label="Generated Spectrum",
        )

        energies, counts = [], []
        if detect_peaks:
            peaks_data = extract_gamma_peaks(
                spectrum,
                calib_coeffs=coeffs,
                min_prominence=min_prominence,
                min_distance=50,
                bounds_path=bounds_path,
            )

            energies = peaks_data["energies"]
            counts = peaks_data["counts"]

            sorted_order = np.argsort(energies)
            energies = energies[sorted_order]
            counts = counts[sorted_order]

            for idx, (p_x, p_y) in enumerate(zip(energies, counts)):
                if coeffs is not None and (p_x < 0.0 or p_x > 3000.0):
                    continue
                y_val = max(p_y, 1e-1)
                stagger_level = idx % 3
                y_text_mult = 2.2 * (2.2**stagger_level)

                ax.annotate(
                    f"{p_x:.1f}",
                    xy=(p_x, y_val * 1.12),
                    xytext=(p_x, y_val * y_text_mult),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    color="darkred",
                    rotation=90,
                    arrowprops=dict(
                        arrowstyle="->",
                        color="red",
                        lw=0.8,
                        shrinkA=0,
                        shrinkB=4,
                    ),
                )

        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-1)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Counts (Log Scale)")

        peak_count_str = len(energies) if detect_peaks else 0
        ax.set_title(
            f"Generated Spectrum {i + 1}/{num_samples} | Model: {model_name} | Detected Peaks: {peak_count_str}"
        )
        ax.grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"spectrum_{i+1:03d}.png"), dpi=300)
        plt.close(fig)

def plot_diffusion_step(
    track_path,
    single_signal,
    noise_scale_val,
    loop_idx,
    label,
    max_val=None,
    calib_params=None,
    bounds_path=None,
):
    """Saves a PNG graph of a single spectrum at a specific step in the diffusion loop."""
    if isinstance(single_signal, torch.Tensor):
        sig_np = single_signal.detach().cpu().numpy()
    else:
        sig_np = single_signal

    L = len(sig_np)

    if max_val is not None:
        sig_np = unnormalize(sig_np, max_val)
        sig_np = np.clip(np.round(sig_np), a_min=0, a_max=None)
        y_label = "Counts (Log Scale)"
        sig_plot = np.maximum(sig_np, 1e-1)
    else:
        y_label = "Amplitude (Normalized)"
        sig_plot = sig_np

    fig, ax = plt.subplots(figsize=(10, 4))

    if calib_params is not None:
        if isinstance(calib_params, torch.Tensor):
            calib_np = calib_params.detach().cpu().numpy()
        else:
            calib_np = np.array(calib_params)

        if bounds_path is None:
            bounds_path = "data/Baq/global_calib_bounds.json"

        try:
            coeffs = unnormalize_coeffs(calib_np, bounds_path=str(bounds_path))
        except Exception:
            coeffs = calib_np

        # Always evaluate energies in keV when calib_params are present
        x_axis = compute_safe_energies(coeffs, L)
        x_label = "Energy (keV)"
        title_ext = " | Calibrated"

        # Enforce X-axis limits: starts at 0.0, ends between 30.0 and 3000.0 keV
        max_e = np.max(x_axis) if len(x_axis) > 0 else 3000.0
        xlim_right = float(np.clip(max_e, 30.0, 3000.0))
        ax.set_xlim(0.0, xlim_right)
    else:
        x_axis = np.arange(L)
        x_label = "Channel"
        title_ext = ""
        ax.set_xlim(0, L)

    ax.plot(x_axis, sig_plot, color="blue", linewidth=1)

    if max_val is not None:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-1)

    ax.set_title(
        f"Diffusion Generation Progress [{str(label).upper()}]{title_ext} | Loop Index: {loop_idx}\nNoise Scale: {noise_scale_val:.6f}"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(track_path, f"{label}_plot.png"), dpi=300)
    plt.close(fig)
