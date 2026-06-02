# Sparsity checks on event histograms from experiment_results.bin and pose estimate extraction
# Saves histogram images to histograms_images/, pose estimates to pose_estimates.txt, and a sparsity plot to sparsity_per_frame.png.
# Runs in folder with experiment_results.bin next to it, run script as is no input arguments 

import struct
import os
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tunable sparsity thresholds
# Set these by inspecting known-good frames first, then bad frames will stand out
SPARSITY_LOW  = 0.01   # below this: too sparse / nothing firing
SPARSITY_HIGH = 0.20   # above this: too dense / noise flooding

# Data characteristics
HIST_ROWS,  HIST_BYTES = 150, 38_400
POS_ROWS,   POS_BYTES  = 150, 12
ATT_ROWS,   ATT_BYTES  = 150, 12

# Read binary file
with open("scripts/experiment/experiment_results.bin", "rb") as f:
    hists     = [np.unpackbits(np.frombuffer(f.read(HIST_BYTES), dtype=np.uint8))
                 .reshape(480, 640) for _ in range(HIST_ROWS)]
    positions = [struct.unpack("<3f", f.read(POS_BYTES)) for _ in range(POS_ROWS)]
    attitudes = [struct.unpack("<3f", f.read(ATT_BYTES)) for _ in range(ATT_ROWS)]

# Save histogram images
hist_dir = "scripts/experiment/histogram_images"
os.makedirs(hist_dir, exist_ok=True)

for i, hist in enumerate(hists):
    img = (hist * 255).astype(np.uint8)
    cv.imwrite(os.path.join(hist_dir, f"hist_{i:03d}.png"), img)

print(f"[INFO] Saved {len(hists)} histogram images to {hist_dir}/")

# Save pose estimates to text file
with open("scripts/experiment/pose_estimates.txt", "w") as f:
    f.write(f"{'Index':<8} {'x':>10} {'y':>10} {'z':>10}    "
            f"{'roll':>10} {'pitch':>10} {'yaw':>10}\n")
    f.write("-" * 72 + "\n")
    for i, (pos, att) in enumerate(zip(positions, attitudes)):
        f.write(f"{i:<8} {pos[0]:>10.4f} {pos[1]:>10.4f} {pos[2]:>10.4f}    "
                f"{att[0]:>10.4f} {att[1]:>10.4f} {att[2]:>10.4f}\n")

print("[INFO] Saved pose estimates to pose_estimates.txt")

# Compute per-frame sparsity and metrics
ratios = np.array([h.mean() for h in hists])
print("\nHistogram Sparsity Check")

mean_r = ratios.mean()
std_r  = ratios.std()
min_r  = ratios.min()
max_r  = ratios.max()

if mean_r > SPARSITY_HIGH:
    flag = "DENSE - threshold may be too low, noise flooding output"
elif mean_r < SPARSITY_LOW:
    flag = "VERY SPARSE - barely any events registered"
else:
    flag = "OK"

print(f"  Active pixel fraction (over all frames):")
print(f"    mean = {mean_r:.4f}   std = {std_r:.4f}")
print(f"    min  = {min_r:.4f}   max = {max_r:.4f}")
print(f"    status: {flag}")

dense_frames  = np.where(ratios > SPARSITY_HIGH)[0]
sparse_frames = np.where(ratios < SPARSITY_LOW)[0]

# if len(dense_frames):
#     print(f"  Dense frames  (>{SPARSITY_HIGH}): {list(dense_frames)}")
# if len(sparse_frames):
#     print(f"  Sparse frames (<{SPARSITY_LOW}):  {list(sparse_frames)}")
# if not len(dense_frames) and not len(sparse_frames):
#     print(f"  All frames within thresholds [{SPARSITY_LOW}, {SPARSITY_HIGH}]")

# Plot: sparsity per frame
fig, ax = plt.subplots(figsize=(12, 4))

frame_indices = np.arange(len(ratios))

colours = np.where(
    ratios > SPARSITY_HIGH, "red",
    np.where(ratios < SPARSITY_LOW, "orange",
    "green")
)
ax.bar(frame_indices, ratios, color=colours, width=0.8)
 
ax.axhline(SPARSITY_HIGH, color="red", linewidth=1.2, linestyle="--",
           label=f"Dense threshold ({SPARSITY_HIGH})")
ax.axhline(SPARSITY_LOW,  color="orange", linewidth=1.2, linestyle="--",
           label=f"Sparse threshold ({SPARSITY_LOW})")
ax.axhline(mean_r, color="black", linewidth=1.0, linestyle=":",
           label=f"Mean ({mean_r:.4f})")

ax.set_xlabel("Histogram index")
ax.set_ylabel("Active pixel fraction")
ax.set_title("Sparsity per Frame")
ax.set_xlim(-0.5, len(ratios) - 0.5)
ax.set_ylim(0, max(ratios.max() * 1.15, SPARSITY_HIGH * 1.2))
ax.legend()

fig.tight_layout()
fig.savefig("scripts/experiment/sparsity_per_frame.png", dpi=150)
plt.close(fig)

print("\n[INFO] Saved sparsity plot to scripts/experiment/sparsity_per_frame.png")

