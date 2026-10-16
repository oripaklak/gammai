#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import runpy
import struct
import numpy as np

# --- DIRECTORY CONFIGURATION ---
BASE_DIR = "results/diff_9M_L36_C128_bs16_cosine-0.01_ep1000_calib_film_Baq/weights-80000_2026-10-04_09-46-24"
OUTPUT_DIR = os.path.join(BASE_DIR, "cnf")

SAMPLES_FILE = os.path.join(BASE_DIR, "all_500_samples.npy")
COEFFS_FILE  = os.path.join(BASE_DIR, "all_500_coeffs.npy")


def load_data(file_path):
    """Loads spectrum data or coefficients from either .npy or .py files."""
    abs_path = os.path.abspath(file_path)

    # Check if alternative extension exists if exact path isn't found
    if not os.path.exists(abs_path):
        base, ext = os.path.splitext(abs_path)
        alt_ext = '.py' if ext == '.npy' else '.npy'
        if os.path.exists(base + alt_ext):
            abs_path = base + alt_ext
        else:
            raise FileNotFoundError(f"Neither {file_path} nor {base + alt_ext} exists.")

    # 1. Try loading as binary .npy array
    try:
        data = np.load(abs_path, allow_pickle=True)
        print(f"Loaded binary NumPy file: {abs_path}")
        return data
    except Exception:
        pass  # Not a valid .npy file, attempt parsing as .py script below

    # 2. Fallback: Run as Python script
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
    """Converts a float to PDP-11 32-bit float format bytes."""
    ieee_bytes = struct.pack('<f', float(val) * 4.0)
    w0, w1 = struct.unpack('<HH', ieee_bytes)
    return struct.pack('<HH', w1, w0)


def encode_time_val(seconds: float) -> bytes:
    """Encodes seconds into Canberra 64-bit inverted 100ns units."""
    ticks = int(seconds * 1e7)
    u64 = (~ticks) & 0xFFFFFFFFFFFFFFFF
    return struct.pack('<Q', u64)


def find_cnf_offsets(buf: bytearray):
    """Locates section offsets inside a CNF binary buffer."""
    offs_param, offs_str, offs_chan = None, None, None
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
        elif sec_id_header == 0x00012001:
            offs_str = sec_loc
        elif sec_id_header == 0x00012005:
            offs_chan = sec_loc

    return offs_param, offs_str, offs_chan


def modify_cnf_template(counts: np.ndarray, template_bytes: bytearray, coeffs=None) -> bytearray:
    """Copies template buffer and updates channel counts and energy coefficients."""
    buf = bytearray(template_bytes)
    offs_param, offs_str, offs_chan = find_cnf_offsets(buf)

    if offs_param is None or offs_chan is None:
        raise ValueError("Invalid CNF template: missing required sections.")

    n_channels = len(counts)
    n_chan_units = (n_channels + 255) // 256
    n_channels_padded = n_chan_units * 256

    if len(counts) < n_channels_padded:
        counts = np.pad(counts, (0, n_channels_padded - len(counts)), mode='constant')

    # Update number of channels multiplier byte
    buf[offs_param + 0x00BA] = n_chan_units & 0xFF

    # Overwrite energy calibration coefficients if provided
    if coeffs is not None and len(coeffs) >= 4:
        offs_calib = offs_param + 0x30 + struct.unpack_from('<H', buf, offs_param + 0x22)[0]
        for idx, c in enumerate(coeffs[:4]):
            buf[offs_calib + 0x44 + (idx * 4): offs_calib + 0x48 + (idx * 4)] = float_to_pdp11f_bytes(c)

    # Overwrite channel counts
    counts_bytes = counts.astype('<u4').tobytes()
    chan_start = offs_chan + 0x200
    chan_end = chan_start + len(counts_bytes)

    if chan_end > len(buf):
        buf.extend(b'\x00' * (chan_end - len(buf)))

    buf[chan_start:chan_end] = counts_bytes
    return buf


def create_cnf_from_scratch(counts: np.ndarray, coeffs=None, sample_name="Generated Spectrum") -> bytearray:
    """Creates a complete CNF binary file from scratch with a valid CAM Header."""
    n_channels = len(counts)
    n_chan_units = (n_channels + 255) // 256
    n_channels_padded = n_chan_units * 256

    if len(counts) < n_channels_padded:
        counts = np.pad(counts, (0, n_channels_padded - len(counts)), mode='constant')

    offs_param, offs_str, offs_mark, offs_chan = 0x200, 0x600, 0xA00, 0xE00
    offs_calib = offs_param + 0x30 + 0x100
    offs_times = offs_param + 0x30 + 0x050

    chan_data_size = 0x200 + n_channels_padded * 4
    total_size = offs_chan + chan_data_size
    buf = bytearray(total_size)

    # =========================================================
    # FIX: Initialize the 112-byte CAM Header (0x00 - 0x6F)
    # =========================================================
    struct.pack_into('<I', buf, 0x00, 0x00000001)  # CAM Revision / Header ID
    buf[0x08:0x0C] = b"PHA "                      # Required "PHA " magic keyword for Genie 2000
    struct.pack_into('<H', buf, 0x10, 0x0070)      # Header length (0x70 = 112 bytes)
    struct.pack_into('<H', buf, 0x12, 0x0004)      # Number of section directory entries

    # Define Section Directory entries starting at 0x70
    sections = [
        (0x00012000, offs_param),
        (0x00012001, offs_str),
        (0x00012004, offs_mark),
        (0x00012005, offs_chan),
        (0x00000000, 0x0)
    ]

    for idx, (sec_id, sec_loc) in enumerate(sections):
        sec_header = 0x70 + idx * 0x30
        struct.pack_into('<I', buf, sec_header, sec_id)
        struct.pack_into('<I', buf, sec_header + 0x0A, sec_loc)
        if sec_id != 0:
            struct.pack_into('<I', buf, sec_loc, sec_id)

    # Configure sub-header offsets & channel unit multiplier
    struct.pack_into('<H', buf, offs_param + 0x22, 0x100)
    struct.pack_into('<H', buf, offs_param + 0x24, 0x050)
    buf[offs_param + 0x00BA] = n_chan_units & 0xFF

    # Energy calibration polynomial coefficients
    if coeffs is None or len(coeffs) < 4:
        coeffs = [0.0, 1.0, 0.0, 0.0]
    for i, c in enumerate(coeffs[:4]):
        buf[offs_calib + 0x44 + (i * 4): offs_calib + 0x48 + (i * 4)] = float_to_pdp11f_bytes(c)

    unit_b = b"keV\x00"
    buf[offs_calib + 0x5C: offs_calib + 0x5C + len(unit_b)] = unit_b
    buf[offs_calib + 0xDC: offs_calib + 0xE0] = float_to_pdp11f_bytes(1.0)

    # Live time and Real time (100.0s default)
    buf[offs_times + 0x09: offs_times + 0x11] = encode_time_val(100.0)
    buf[offs_times + 0x11: offs_times + 0x19] = encode_time_val(100.0)

    # Sample Name string
    s_bytes = sample_name.encode('utf-8')[:0x40]
    buf[offs_str + 0x0030: offs_str + 0x0030 + len(s_bytes)] = s_bytes

    # Channel counts array
    counts_bytes = counts.astype('<u4').tobytes()
    buf[offs_chan + 0x200: offs_chan + 0x200 + len(counts_bytes)] = counts_bytes

    return buf


def main(template_path=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load spectrum samples
    samples_data = load_data(SAMPLES_FILE)

    # Load coefficients if available
    coeffs_data = None
    try:
        coeffs_data = load_data(COEFFS_FILE)
    except Exception as e:
        print(f"Notice: Proceeding without coefficients file ({e}).")

    # Load optional template file
    template_bytes = None
    if template_path and os.path.exists(template_path):
        with open(template_path, 'rb') as f:
            template_bytes = bytearray(f.read())
        print(f"Loaded CNF template: {template_path}")

    print(f"Converting {len(samples_data)} spectra to CNF...")

    for i, spec in enumerate(samples_data):
        counts = np.array(spec, dtype=np.uint32)
        coeffs = coeffs_data[i] if (coeffs_data is not None and i < len(coeffs_data)) else None

        output_path = os.path.join(OUTPUT_DIR, f"spectrum_{i+1:02d}.CNF")

        if template_bytes is not None:
            cnf_buf = modify_cnf_template(counts, template_bytes, coeffs=coeffs)
        else:
            cnf_buf = create_cnf_from_scratch(counts, coeffs=coeffs, sample_name=f"Generated Spectrum {i+1}")

        with open(output_path, "wb") as f:
            f.write(cnf_buf)

        print(f"Saved: {output_path}")

    print(f"\nAll {len(samples_data)} .CNF files generated in:\n{os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    template_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(template_file)