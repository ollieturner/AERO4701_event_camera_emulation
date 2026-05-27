import struct
import numpy as np
import cv2 as cv
import os

# Data characteristics
HIST_ROWS, HIST_BYTES = 150, 38_400
POS_ROWS,  POS_BYTES  = 150, 12
ATT_ROWS,  ATT_BYTES  = 150, 12

# Read binary file
with open("experiment_results.bin", "rb") as f:
    hists     = [np.unpackbits(np.frombuffer(f.read(HIST_BYTES), dtype=np.uint8))
                 .reshape(480, 640) for _ in range(HIST_ROWS)]
    positions = [struct.unpack("<3f", f.read(POS_BYTES)) for _ in range(POS_ROWS)]
    attitudes = [struct.unpack("<3f", f.read(ATT_BYTES)) for _ in range(ATT_ROWS)]

# Make folder to save histogram photos
hist_dir = "histograms"
os.makedirs(hist_dir, exist_ok=True)

# Convert histogram data to image for visualisation
for i, hist in enumerate(hists):
    img = (hist * 255).astype(np.uint8)
    cv.imwrite(os.path.join(hist_dir, f"hist_{i:03d}.png"), img)

for pos, att in zip(positions, attitudes):
    # pos: (x, y, z)   att: (roll, pitch, yaw)
    print(f"pos={pos}  att={att}\n")
