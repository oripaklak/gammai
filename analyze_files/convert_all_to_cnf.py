#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import runpy
import struct
import numpy as np

# --- DIRECTORY CONFIGURATION ---
BASE_DIR = "results/diff_9M_L36_C128_bs16_cosine-0.01_ep1000_calib_film_Baq/weights-80000_2026-10-04_09-25-41"
OUTPUT_DIR = os.path.join(BASE_DIR, "cnf")

SAMPLES_FILE = os.path.join(BASE_DIR, "all_500_samples.npy")
COEFFS_FILE  = os.path.join(BASE_DIR, "all_500_coeffs.npy")

# Set default template to your verified original CNF
DEFAULT_TEMPLATE = "analyze_files/1-50-bge-010126.CNF"


def load_data(file_path):
    """Loads spectrum data or coefficients from either .npy or .py files."""
    abs_path = os.path.abspath(file_path)

    if not os.path.exists(abs_path):
        base, ext = os.path.splitext(abs_path)
        alt_ext = '.py' if ext == '.npy' else '.npy'
        if os.path.exists(base + alt_ext):
            abs_path = base + alt_ext
        else:
            raise FileNotFoundError(f"Neither {file_path} nor {base + alt_ext} exists.")

    try:
        data = np.load(abs_path, allow_pickle=True)
        print(f"Loaded binary NumPy file: {abs_path}")
        return data
    except Exception:
        pass

    try:
        env = runpy.run_path(abs_path)
        for var_name in ['samples', 'coeffs', 'data', 'all_samples', 'all_coeffs', 'arr']:
            if var_name in env:
                print(f"Loaded variable '{var_name}' from Python script: {abs_path}")
                return env[var_name]

        for k, v in env.items():
            if not k.startswith("__") and isinstance(v, (list, np.ndarray)):
                print(f"Loaded array '{k}' from Python script: {abs_path}")
                return v
    except Exception as e:
        raise ValueError(f"Failed to load {abs_path}: {e}")

    raise ValueError(f"Could not extract array from {abs_path}")


def float_to_pdp11f_bytes(val: float) -> bytes:
    """Converts a float to PDP-11 32-bit floating point format bytes."""
    val = float(val)
    if val == 0.0:
        return b'\x00\x00\x00\x00'
    ieee_bytes = struct.pack('<f', val * 4.0)
    w0, w1 = struct.unpack('<HH', ieee_bytes)
    return struct.pack('<HH', w1, w0)


def find_cnf_offsets(buf: bytearray):
    """Locates section offsets inside a CNF binary buffer via CAM directory table."""
    offs_param, offs_chan = None, None
    i = 0
    while True:
        sec_header = 0x70 + i * 0x30
        i += 1
        if sec_header + 0x0E > len(buf):
            break
        sec_id_header = struct.unpack_from('<I', buf, sec_header)[0]
        if sec_id_header == 0x00:
            break
        sec_loc = struct.unpack_from('<I', buf, sec_header + 0x0A)[0]

        if sec_id_header == 0x00012000:
            offs_param = sec_loc
        elif sec_id_header == 0x00012005:
            offs_chan = sec_loc

    return offs_param, offs_chan


def modify_cnf_template(counts: np.ndarray, template_bytes: bytearray, coeffs=None) -> bytearray:
    """
    Copies the original template buffer and safely injects:
      1. New channel counts in section 0x00012005
      2. Energy calibration polynomial coefficients in section 0x00012000
    All other section structures, headers, and metadata remain untouched.
    """
    buf = bytearray(template_bytes)
    offs_param, offs_chan = find_cnf_offsets(buf)

    if offs_param is None or offs_chan is None:
        raise ValueError("Invalid CNF template: missing section 0x00012000 or 0x00012005.")

    # Read declared channel count from template: uint8 multiplier at offs_param + 0xBA
    template_n_chan = buf[offs_param + 0x00BA] * 256
    if template_n_chan == 0:
        template_n_chan = len(counts)

    # Align input counts to template channel length
    counts = np.asarray(counts, dtype=np.uint32)
    if len(counts) < template_n_chan:
        counts = np.pad(counts, (0, template_n_chan - len(counts)), mode='constant')
    elif len(counts) > template_n_chan:
        counts = counts[:template_n_chan]

    # 1. Update Energy Calibration Coefficients (if supplied)
    if coeffs is not None and len(coeffs) >= 4:
        sub_record_rel = struct.unpack_from('<H', buf, offs_param + 0x22)[0]
        offs_calib = offs_param + 0x30 + sub_record_rel
        for idx, c in enumerate(coeffs[:4]):
            buf[offs_calib + 0x44 + (idx * 4) : offs_calib + 0x48 + (idx * 4)] = float_to_pdp11f_bytes(c)

    # 2. Update Channel Data in section 0x00012005
    counts_bytes = counts.astype('<u4').tobytes()
    chan_start = offs_chan + 0x200
    chan_end = chan_start + len(counts_bytes)

    # Write without changing total file length or corrupting subsequent sections
    buf[chan_start:chan_end] = counts_bytes

    return buf


def main(template_path=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine template file
    tpl_file = template_path or DEFAULT_TEMPLATE
    if not os.path.exists(tpl_file):
        raise FileNotFoundError(
            f"CNF template file not found at '{tpl_file}'. "
            f"Please place an original valid CNF there or pass the path as an argument."
        )

    with open(tpl_file, 'rb') as f:
        template_bytes = bytearray(f.read())
    print(f"Loaded CNF template: {tpl_file} ({len(template_bytes):,} bytes)")

    # Load spectrum samples
    samples_data = load_data(SAMPLES_FILE)

    # Load coefficients if available
    coeffs_data = None
    try:
        coeffs_data = load_data(COEFFS_FILE)
    except Exception as e:
        print(f"Notice: Proceeding without coefficients file ({e}).")

    print(f"Converting {len(samples_data)} spectra to CNF using template...")

    for i, spec in enumerate(samples_data):
        counts = np.array(spec, dtype=np.uint32)
        coeffs = coeffs_data[i] if (coeffs_data is not None and i < len(coeffs_data)) else None

        output_path = os.path.join(OUTPUT_DIR, f"spectrum_{i+1:02d}.CNF")
        cnf_buf = modify_cnf_template(counts, template_bytes, coeffs=coeffs)

        with open(output_path, "wb") as f:
            f.write(cnf_buf)

    print(f"\nAll {len(samples_data)} .CNF files successfully generated in:\n{os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    template_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(template_file)