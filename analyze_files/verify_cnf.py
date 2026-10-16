import struct
import sys

def verify_cnf(file_path: str):
    """Verifies if a CNF file has the required CAM Header and sections for Genie 2000."""
    with open(file_path, 'rb') as f:
        buf = f.read()

    # 1. Header size check
    if len(buf) < 112:
        return False, "File is corrupted or too small (< 112 bytes)."

    # 2. Header emptiness check
    if buf[:112] == b'\x00' * 112:
        return False, "Header (first 112 bytes) is empty (all zeros)."

    # 3. PHA Magic Keyword check
    magic = buf[0x08:0x0C]
    if magic != b"PHA ":
        return False, f"Missing 'PHA ' keyword at offset 0x08 (found {magic})."

    # 4. Section Directory check
    offs_param, offs_chan = None, None
    i = 0
    while True:
        sec_header = 0x70 + i * 0x30
        i += 1
        if sec_header + 0x0E > len(buf):
            break
        sec_id = struct.unpack_from('<I', buf, sec_header)[0]
        if sec_id == 0x00:
            break
        sec_loc = struct.unpack_from('<I', buf, sec_header + 0x0A)[0]

        if sec_id == 0x00012000:
            offs_param = sec_loc
        elif sec_id == 0x00012005:
            offs_chan = sec_loc

    if offs_param is None or offs_chan is None:
        return False, "Missing required parameter (0x12000) or channel (0x12005) sections."

    return True, f"Valid CNF structure! Total size: {len(buf)} bytes."

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "output.CNF"
    is_valid, message = verify_cnf(filepath)
    print(f"Validation Result: {'[PASS]' if is_valid else '[FAIL]'} - {message}")