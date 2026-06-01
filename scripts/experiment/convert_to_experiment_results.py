"""
convert_to_experiment_results.py

Converts a legacy experiment folder containing:
  - camera_attitude.csv  (CSV: roll,pitch,yaw — no header)
  - camera_position.csv  (CSV: x,y,z — no header)
  - histograms/hist_001.npy ... hist_099.npy  (packed 1-bit uint8 arrays)

into a single binary experiment_results.bin with the layout:

  150 histogram rows × 38,400 bytes  (640×480×1 bit packed, zero-padded)
  150 pose rows      × 24 bytes      (x,y,z,roll,pitch,yaw as 6× float32 LE, zero-padded)

  Total: 5,763,600 bytes

Usage:
    python convert_to_experiment_results.py <input_folder> [output_file]
"""

# in vscode run with: .\scripts\experiment\convert_to_experiment_results.py .\scripts\experiment\legacy_data\ experiment_results.bin

import sys
import os
import csv
import glob
import struct
import numpy as np

HIST_ROWS      = 150
POSE_ROWS      = 150
HIST_ROW_BYTES = 38_400
POSE_ROW_BYTES = 24
ZERO_HIST      = b"\x00" * HIST_ROW_BYTES
ZERO_POSE      = struct.pack("<6f", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

HIST_ROW_BYTES = 38_400
POS_ROW_BYTES  = 12    # 3× float32
ATT_ROW_BYTES  = 12    # 3× float32

def load_csv_no_header(path, num_cols):
    rows = []
    with open(path, newline="") as f:
        for lineno, row in enumerate(csv.reader(f), start=1):
            row = [c.strip() for c in row if c.strip()]
            if not row:
                continue
            if len(row) != num_cols:
                raise ValueError(
                    f"{path} line {lineno}: expected {num_cols} values, got {len(row)}: {row}"
                )
            rows.append([float(v) for v in row])
    return rows


def pack_histogram(npy_path):
    data = np.load(npy_path).tobytes()
    if len(data) < HIST_ROW_BYTES:
        data = data + b"\x00" * (HIST_ROW_BYTES - len(data))
    return data[:HIST_ROW_BYTES]


def convert(input_folder, output_file):
    position_path = os.path.join(input_folder, "camera_position.csv")
    attitude_path = os.path.join(input_folder, "camera_attitude.csv")
    hist_dir      = os.path.join(input_folder, "histograms")

    for p in (position_path, attitude_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Cannot find: {p}")

    position_rows = load_csv_no_header(position_path, 3)
    attitude_rows = load_csv_no_header(attitude_path, 3)

    if len(position_rows) != len(attitude_rows):
        raise ValueError(
            f"Row count mismatch: {len(position_rows)} position rows "
            f"vs {len(attitude_rows)} attitude rows."
        )
    print(f"Loaded {len(position_rows)} pose rows.")

    if not os.path.isdir(hist_dir):
        raise FileNotFoundError(f"Histogram directory not found: {hist_dir}")
    hist_files = sorted(glob.glob(os.path.join(hist_dir, "hist_*.npy")))
    print(f"Loaded {len(hist_files)} histogram files.")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with open(output_file, "wb") as out:
        # Histogram rows (150 × 38,400 bytes)
        for i in range(HIST_ROWS):
            out.write(pack_histogram(hist_files[i]) if i < len(hist_files) else ZERO_HIST)
        # Position rows (150 × 12 bytes: x, y, z as 3× float32)
        for i in range(POSE_ROWS):
            if i < len(position_rows):
                out.write(struct.pack("<3f", *position_rows[i]))
            else:
                out.write(struct.pack("<3f", 0.0, 0.0, 0.0))
        # Attitude rows (150 × 12 bytes: r, p, y as 3× float32)
        for i in range(POSE_ROWS):
            if i < len(attitude_rows):
                out.write(struct.pack("<3f", *attitude_rows[i]))
            else:
                out.write(struct.pack("<3f", 0.0, 0.0, 0.0))
    
    total = HIST_ROWS * HIST_ROW_BYTES + POSE_ROWS * POSE_ROW_BYTES
    print(f"Written: {output_file}  ({total:,} bytes / {total/1024/1024:.2f} MB)")



def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    input_folder = sys.argv[1]
    if not os.path.isdir(input_folder):
        print(f"Error: '{input_folder}' is not a directory.")
        sys.exit(1)
    output_file = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(input_folder, "experiment_results.bin")
    try:
        convert(input_folder, output_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

