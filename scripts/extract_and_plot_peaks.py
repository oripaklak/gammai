import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import helpers
from utils import plot 

def load_spectrum_txt(txt_path: Path) -> np.ndarray:
    """Loads a 1D counts array from a tab-separated 'Channel\\tCounts' .txt file."""
    data = np.loadtxt(txt_path, delimiter="\t", skiprows=1, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected 2 columns (Channel, Counts) in {txt_path}, got shape {data.shape}"
        )

    channels = data[:, 0].astype(int)
    counts = data[:, 1]

    if np.array_equal(channels, np.arange(len(channels))):
        return counts

    spectrum = np.zeros(int(channels.max()) + 1, dtype=float)
    spectrum[channels] = counts
    return spectrum


def load_calib_coeffs_txt(coeff_path: Path) -> np.ndarray:
    """Loads polynomial calibration coefficients from a 'c0: <val>' style .txt file."""
    coeffs_by_idx = {}
    pattern = re.compile(r"^c(\d+)\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")

    with open(coeff_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            match = pattern.match(stripped)
            if match:
                coeffs_by_idx[int(match.group(1))] = float(match.group(2))

    if not coeffs_by_idx:
        raise ValueError(f"No calibration coefficients (c0, c1, ...) found in: {coeff_path}")

    return np.array([coeffs_by_idx[i] for i in sorted(coeffs_by_idx)], dtype=float)


def resolve_coeff_path(spectrum_path: Path, explicit_coeff_file: str | None) -> Path | None:
    """Finds the calibration coefficient file explicitly or by mapping spectra_txt -> coeff_txt."""
    if explicit_coeff_file is not None:
        p = Path(explicit_coeff_file)
        if not p.is_absolute() and not p.exists():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Calibration coefficient file not found: {p}")
        return p

    parts = list(spectrum_path.parts)
    if "spectra_txt" in parts:
        idx = parts.index("spectra_txt")
        parts[idx] = "coeff_txt"
        inferred = Path(*parts)
        if inferred.exists():
            return inferred

    fallback = PROJECT_ROOT / "data" / "BAQ" / "coeff_txt" / spectrum_path.name
    if fallback.exists():
        return fallback

    return None


def resolve_nudat_path(explicit_nudat_file: str | None) -> Path | None:
    """Finds nudat_gamma.json explicitly or in common project locations."""
    if explicit_nudat_file is not None:
        p = Path(explicit_nudat_file)
        if not p.is_absolute() and not p.exists():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"NuDat JSON file not found: {p}")
        return p

    candidates = [
        PROJECT_ROOT / "nudat_gamma.json",
        PROJECT_ROOT / "data" / "nudat_gamma.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract gamma photopeaks, identify isotopes, and plot spectrum from a .txt file."
    )
    parser.add_argument(
        "spectrum_file",
        type=str,
        nargs="?",
        default="data/BAQ/spectra_txt/Y88-CMI2020-5cm-150321-50.txt",
        help="Path to the input .txt spectrum file.",
    )
    parser.add_argument(
        "--coeff-file",
        type=str,
        default=None,
        help="Path to calibration coeff .txt file (auto-detected from coeff_txt/ if omitted).",
    )
    parser.add_argument(
        "--nudat-file",
        type=str,
        default=None,
        help="Path to nudat_gamma.json (auto-detected if omitted).",
    )
    parser.add_argument(
        "--calib-coeffs",
        type=float,
        nargs="+",
        default=None,
        help="Manual calibration coefficients override.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT),
        help="Directory to save outputs (defaults to gammai root directory).",
    )
    parser.add_argument("--bounds-path", type=str, default=None)
    parser.add_argument("--fwhm", type=float, default=8.0, help="Estimated FWHM in channels.")
    parser.add_argument("--confidence-factor", type=float, default=3.5)
    parser.add_argument("--min-distance", type=int, default=20)
    parser.add_argument("--min-energy", type=float, default=40.0)
    parser.add_argument("--smoothing-iterations", type=int, default=5)
    parser.add_argument(
        "--intensity-threshold",
        type=float,
        default=10.0,
        help="Minimum NuDat intensity (%) threshold T for isotope matching.",
    )
    parser.add_argument(
        "--energy-tol",
        type=float,
        default=1.5,
        help="Energy matching tolerance in keV.",
    )
    parser.add_argument(
        "--allow-partial-lines",
        action="store_true",
        help="Allow isotope match even if weaker lines above T are not all detected.",
    )
    args = parser.parse_args()

    input_path = Path(args.spectrum_file)
    if not input_path.is_absolute() and not input_path.exists():
        input_path = PROJECT_ROOT / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Spectrum file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load spectrum
    spectrum = load_spectrum_txt(input_path)
    spectra_batch = spectrum[np.newaxis, :]

    # 2. Load calibration coefficients
    gen_coeffs = None
    if args.calib_coeffs is not None:
        coeffs_1d = np.asarray(args.calib_coeffs, dtype=float)
        gen_coeffs = coeffs_1d[np.newaxis, :]
    else:
        coeff_path = resolve_coeff_path(input_path, args.coeff_file)
        if coeff_path is not None:
            coeffs_1d = load_calib_coeffs_txt(coeff_path)
            gen_coeffs = coeffs_1d[np.newaxis, :]
            print(f"Loaded calibration coefficients from {coeff_path}: {coeffs_1d}")
        else:
            print("No calibration file found; running in Channel space.")

    # 3. Load NuDat gamma database
    nudat_path = resolve_nudat_path(args.nudat_file)
    nudat_db = helpers.load_isotope_dict(nudat_path) if nudat_path is not None else None
    if nudat_path is not None:
        print(f"Loaded NuDat database from {nudat_path}")
    else:
        print("nudat_gamma.json not found; skipping isotope identification.")

    require_all = not args.allow_partial_lines
    # --------- PARAMETERS ------------
    min_prominence = 3.5
    min_distance = 20 # measured in channels
    window_len = 13 # 1.5 x FWHM, e.g: 11, 13, 15
    # ---------------------------------
    
    # 4. Extract peaks + isotopes and save CSV/TXT
    isotope_groups_list = helpers.save_extracted_peaks_to_files(
        spectra_batch,
        save_dir=str(output_dir),
        gen_coeffs=gen_coeffs,
        min_prominence=min_prominence,
        min_distance=min_distance,
        window_len=window_len,
        bounds_path=args.bounds_path,
    )

    stem = input_path.stem
    for default_name, target_name in [
        ("extracted_peaks.csv", f"{stem}_peaks.csv"),
        ("extracted_peaks.txt", f"{stem}_peaks.txt"),
    ]:
        src = output_dir / default_name
        dst = output_dir / target_name
        if src.exists():
            src.replace(dst)

        # Identify isotopes from the saved peaks CSV
        peaks_csv_path = output_dir / f"{stem}_peaks.csv"
        if nudat_db is not None and gen_coeffs is not None and peaks_csv_path.exists():
            isotope_dict = helpers.extract_isotopes(
                peaks_csv_path,
                nudat_db,  # reuses the dict already loaded in step 3 (nudat_path also works)
                intensity_threshold=args.intensity_threshold,
                energy_tol=args.energy_tol,
                min_energy=args.min_energy,
                require_all=require_all,
            )
        else:
            isotope_dict = {}

        # Save isotope dictionary JSON: {<isotope>: <energies>, "doesn't belong to any isotopes": <energies>}
        iso_json_path = output_dir / f"{stem}_isotopes.json"
        with open(iso_json_path, "w", encoding="utf-8") as jf:
            json.dump(isotope_dict, jf, indent=2)

        print("Extracted Isotopes:", json.dumps(isotope_dict, indent=2))

    # 5. Plot and save spectrum with energy + isotope annotations
    plot.plot_and_save_spectra(
        spectra_batch,
        str(output_dir),
        stem,
        min_prominence=min_prominence,
        min_distance=min_distance,
        window_len=window_len,
        gen_coeffs=gen_coeffs,
        detect_peaks=True,
    )

    default_plot = output_dir / "spectrum_001.png"
    target_plot = output_dir / f"{stem}_spectrum.png"
    if default_plot.exists():
        default_plot.replace(target_plot)


if __name__ == "__main__":
    main()