"""
verify_histograms.py
Level 1a sparsity checks on event histograms from experiment_results.bin.
Saves histogram images to histograms_images/, pose estimates to pose_estimates.txt,
and a sparsity plot to sparsity_per_frame.png.
"""

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
hist_dir = "scripts/experiment/histograms_images"
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

# Compute per-frame sparsity
ratios = np.array([h.mean() for h in hists])

# Level 1a: Sparsity
print("\nHistogram Sparsity Check")

mean_r = ratios.mean()
std_r  = ratios.std()
min_r  = ratios.min()
max_r  = ratios.max()

if mean_r > SPARSITY_HIGH:
    flag = "DENSE - emulator threshold may be too low, noise flooding output"
elif mean_r < SPARSITY_LOW:
    flag = "VERY SPARSE - barely any events registered, check emulator sensitivity"
else:
    flag = "OK"

print(f"  Active pixel fraction (all frames):")
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






# """
# verify_histograms.py
# --------------------
# Level 1 sanity checks on event histograms from experiment_results.bin.
# Saves histogram images to histograms/, pose estimates to pose_estimates.txt,
# and a sparsity plot to sparsity_per_frame.png.
# """

# import struct
# import os
# import numpy as np
# import cv2 as cv
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# # ── Tunable sparsity thresholds ───────────────────────────────────────────────
# # Set these by inspecting known-good frames first, then bad frames will stand out
# SPARSITY_LOW  = 0.01   # below this → too sparse / nothing firing
# SPARSITY_HIGH = 0.40   # above this → too dense / noise flooding

# # ── Data characteristics ──────────────────────────────────────────────────────
# HIST_ROWS,  HIST_BYTES = 150, 38_400
# POS_ROWS,   POS_BYTES  = 150, 12
# ATT_ROWS,   ATT_BYTES  = 150, 12

# # ── Read binary file ──────────────────────────────────────────────────────────
# with open("scripts/experiment/experiment_results.bin", "rb") as f:
#     hists     = [np.unpackbits(np.frombuffer(f.read(HIST_BYTES), dtype=np.uint8))
#                  .reshape(480, 640) for _ in range(HIST_ROWS)]
#     positions = [struct.unpack("<3f", f.read(POS_BYTES)) for _ in range(POS_ROWS)]
#     attitudes = [struct.unpack("<3f", f.read(ATT_BYTES)) for _ in range(ATT_ROWS)]

# # ── Save histogram images ─────────────────────────────────────────────────────
# hist_dir = "scripts/experiment/histograms_images"
# os.makedirs(hist_dir, exist_ok=True)

# for i, hist in enumerate(hists):
#     img = (hist * 255).astype(np.uint8)
#     cv.imwrite(os.path.join(hist_dir, f"hist_{i:03d}.png"), img)

# print(f"[INFO] Saved {len(hists)} histogram images to {hist_dir}/")

# # ── Save pose estimates to text file ─────────────────────────────────────────
# with open("scripts/experiment/pose_estimates.txt", "w") as f:
#     f.write(f"{'Index':<8} {'x':>10} {'y':>10} {'z':>10}    "
#             f"{'roll':>10} {'pitch':>10} {'yaw':>10}\n")
#     f.write("-" * 72 + "\n")
#     for i, (pos, att) in enumerate(zip(positions, attitudes)):
#         f.write(f"{i:<8} {pos[0]:>10.4f} {pos[1]:>10.4f} {pos[2]:>10.4f}    "
#                 f"{att[0]:>10.4f} {att[1]:>10.4f} {att[2]:>10.4f}\n")

# print("[INFO] Saved pose estimates to pose_estimates.txt")

# # ── Compute per-frame sparsity ────────────────────────────────────────────────
# ratios = np.array([h.mean() for h in hists])

# # ── Level 1a: Sparsity (aggregate) ───────────────────────────────────────────
# print("\n── Level 1a: Sparsity ──────────────────────────────────────────────────")

# mean_r = ratios.mean()
# std_r  = ratios.std()
# min_r  = ratios.min()
# max_r  = ratios.max()

# if mean_r > SPARSITY_HIGH:
#     flag = "⚠  DENSE — emulator threshold may be too low, noise flooding output"
# elif mean_r < SPARSITY_LOW:
#     flag = "⚠  VERY SPARSE — barely any events registered, check emulator sensitivity"
# else:
#     flag = "✓  OK"

# print(f"  Active pixel fraction (all frames):")
# print(f"    mean = {mean_r:.4f}   std = {std_r:.4f}")
# print(f"    min  = {min_r:.4f}   max = {max_r:.4f}")
# print(f"    {flag}")

# # Per-frame outliers against the tunable thresholds
# dense_frames   = np.where(ratios > SPARSITY_HIGH)[0]
# sparse_frames  = np.where(ratios < SPARSITY_LOW)[0]

# if len(dense_frames):
#     print(f"  ⚠  Dense frames  (>{SPARSITY_HIGH}): {list(dense_frames)}")
# if len(sparse_frames):
#     print(f"  ⚠  Sparse frames (<{SPARSITY_LOW}):  {list(sparse_frames)}")
# if not len(dense_frames) and not len(sparse_frames):
#     print(f"  ✓  All frames within thresholds [{SPARSITY_LOW}, {SPARSITY_HIGH}]")

# # ── Level 1b: Temporal Consistency ───────────────────────────────────────────
# print("\n── Level 1b: Temporal Consistency ──────────────────────────────────────")

# counts    = ratios * (480 * 640)
# deltas    = np.abs(np.diff(counts))
# mean_d    = deltas.mean()
# std_d     = deltas.std()
# spike_thr = mean_d + 4 * std_d
# spikes    = int((deltas > spike_thr).sum())
# stagnant  = int((deltas < 0.01 * counts.mean()).sum())

# spike_flag    = f"⚠  {spikes} spike(s) detected" if spikes   > 0 else "✓  No spikes"
# stagnant_flag = f"⚠  {stagnant} near-identical consecutive pair(s)" if stagnant > 0 else "✓  No stagnant pairs"

# print(f"  Active pixel count Δ per histogram:")
# print(f"    mean = {mean_d:.1f}   std = {std_d:.1f}")
# print(f"    {spike_flag}")
# print(f"    {stagnant_flag}")

# # ── Level 1c: Spatial Uniformity ─────────────────────────────────────────────
# print("\n── Level 1c: Spatial Uniformity (4×4 grid) ─────────────────────────────")

# accum     = sum(h.astype(np.float32) for h in hists)
# H, W      = accum.shape
# gh, gw    = H // 4, W // 4
# grid      = np.array([[accum[r*gh:(r+1)*gh, c*gw:(c+1)*gw].sum()
#                         for c in range(4)] for r in range(4)])
# grid_norm = grid / grid.sum()
# max_cell  = grid_norm.max()
# min_cell  = grid_norm.min()

# uniform_flag = "⚠  BIASED — activity concentrated in one region" if max_cell > 0.15 else "✓  OK"

# print(f"  Cell activity fractions (expected ≈ 0.0625 each):")
# for row in grid_norm:
#     print("    " + "  ".join(f"{v:.4f}" for v in row))
# print(f"    max = {max_cell:.4f}   min = {min_cell:.4f}")
# print(f"    {uniform_flag}")

# # ── Summary ───────────────────────────────────────────────────────────────────
# print("\n── Summary ──────────────────────────────────────────────────────────────")
# print(f"  Sparsity         {flag}")
# print(f"  Temporal spikes  {spike_flag}")
# print(f"  Stagnant pairs   {stagnant_flag}")
# print(f"  Spatial bias     {uniform_flag}")

# # ── Plot: sparsity per frame ──────────────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(12, 4))

# frame_indices = np.arange(len(ratios))

# # Colour each bar by whether it is within thresholds
# colours = np.where(
#     ratios > SPARSITY_HIGH, "#e74c3c",        # red   — too dense
#     np.where(ratios < SPARSITY_LOW, "#e67e22", # orange — too sparse
#     "#2ecc71")                                  # green  — OK
# )
# ax.bar(frame_indices, ratios, color=colours, width=0.8, zorder=2)

# # Threshold bands
# ax.axhline(SPARSITY_HIGH, color="#e74c3c", linewidth=1.2, linestyle="--",
#            label=f"Dense threshold ({SPARSITY_HIGH})")
# ax.axhline(SPARSITY_LOW,  color="#e67e22", linewidth=1.2, linestyle="--",
#            label=f"Sparse threshold ({SPARSITY_LOW})")

# # Overall mean
# ax.axhline(mean_r, color="white", linewidth=1.0, linestyle=":",
#            label=f"Mean ({mean_r:.4f})")

# ax.set_xlabel("Histogram index", fontsize=11)
# ax.set_ylabel("Active pixel fraction", fontsize=11)
# ax.set_title("Sparsity per frame", fontsize=13)
# ax.set_xlim(-0.5, len(ratios) - 0.5)
# ax.set_ylim(0, max(ratios.max() * 1.15, SPARSITY_HIGH * 1.2))
# ax.legend(fontsize=9)
# ax.set_facecolor("#1a1a2e")
# fig.patch.set_facecolor("#16213e")
# ax.tick_params(colors="white")
# ax.xaxis.label.set_color("white")
# ax.yaxis.label.set_color("white")
# ax.title.set_color("white")
# for spine in ax.spines.values():
#     spine.set_edgecolor("#444")

# fig.tight_layout()
# fig.savefig("scripts/experiment/sparsity_per_frame.png", dpi=150, facecolor=fig.get_facecolor())
# plt.close(fig)

# print("\n[INFO] Saved sparsity plot to sparsity_per_frame.png")





# # Sparsity check
# """
# verify_histograms.py
# --------------------
# Level 1 sanity checks on event histograms from experiment_results.bin.
# Saves histogram images to histograms/ and pose estimates to pose_estimates.txt.
# """

# import struct
# import os
# import numpy as np
# import cv2 as cv

# # ── Data characteristics ──────────────────────────────────────────────────────
# HIST_ROWS,  HIST_BYTES = 150, 38_400
# POS_ROWS,   POS_BYTES  = 150, 12
# ATT_ROWS,   ATT_BYTES  = 150, 12

# # ── Read binary file ──────────────────────────────────────────────────────────
# with open("scripts/experiment/experiment_results.bin", "rb") as f:
#     hists     = [np.unpackbits(np.frombuffer(f.read(HIST_BYTES), dtype=np.uint8))
#                  .reshape(480, 640) for _ in range(HIST_ROWS)]
#     positions = [struct.unpack("<3f", f.read(POS_BYTES)) for _ in range(POS_ROWS)]
#     attitudes = [struct.unpack("<3f", f.read(ATT_BYTES)) for _ in range(ATT_ROWS)]

# # ── Save histogram images ─────────────────────────────────────────────────────
# hist_dir = "scripts/experiment/histogram_images"
# os.makedirs(hist_dir, exist_ok=True)

# for i, hist in enumerate(hists):
#     img = (hist * 255).astype(np.uint8)
#     cv.imwrite(os.path.join(hist_dir, f"hist_{i:03d}.png"), img)

# print(f"[INFO] Saved {len(hists)} histogram images to {hist_dir}/")

# # ── Save pose estimates to text file ─────────────────────────────────────────
# with open("scripts/experiment/pose_estimates.txt", "w") as f:
#     f.write(f"{'Index':<8} {'x':>10} {'y':>10} {'z':>10}    "
#             f"{'roll':>10} {'pitch':>10} {'yaw':>10}\n")
#     f.write("-" * 72 + "\n")
#     for i, (pos, att) in enumerate(zip(positions, attitudes)):
#         f.write(f"{i:<8} {pos[0]:>10.4f} {pos[1]:>10.4f} {pos[2]:>10.4f}    "
#                 f"{att[0]:>10.4f} {att[1]:>10.4f} {att[2]:>10.4f}\n")

# print("[INFO] Saved pose estimates to pose_estimates.txt")

# # ── Level 1a: Sparsity ────────────────────────────────────────────────────────
# print("\n── Level 1a: Sparsity ──────────────────────────────────────────────────")

# ratios = np.array([h.mean() for h in hists])
# mean_r = ratios.mean()
# std_r  = ratios.std()
# min_r  = ratios.min()
# max_r  = ratios.max()

# if mean_r > 0.40:
#     flag = "⚠  DENSE — emulator threshold may be too low, noise flooding output"
# elif mean_r < 0.01:
#     flag = "⚠  VERY SPARSE — barely any events registered, check emulator sensitivity"
# else:
#     flag = "✓  OK"

# print(f"  Active pixel fraction:")
# print(f"    mean = {mean_r:.4f}   std = {std_r:.4f}")
# print(f"    min  = {min_r:.4f}   max = {max_r:.4f}")
# print(f"    {flag}")

# # ── Level 1b: Temporal Consistency ───────────────────────────────────────────
# print("\n── Level 1b: Temporal Consistency ──────────────────────────────────────")

# counts    = np.array([h.sum() for h in hists], dtype=np.float32)
# deltas    = np.abs(np.diff(counts))
# mean_d    = deltas.mean()
# std_d     = deltas.std()
# spike_thr = mean_d + 4 * std_d
# spikes    = int((deltas > spike_thr).sum())
# stagnant  = int((deltas < 0.01 * counts.mean()).sum())

# spike_flag    = f"⚠  {spikes} spike(s) detected" if spikes    > 0 else "✓  No spikes"
# stagnant_flag = f"⚠  {stagnant} near-identical consecutive pair(s)" if stagnant > 0 else "✓  No stagnant pairs"

# print(f"  Active pixel count Δ per histogram:")
# print(f"    mean = {mean_d:.1f}   std = {std_d:.1f}")
# print(f"    {spike_flag}")
# print(f"    {stagnant_flag}")

# # ── Level 1c: Spatial Uniformity ─────────────────────────────────────────────
# print("\n── Level 1c: Spatial Uniformity (4×4 grid) ─────────────────────────────")

# accum     = sum(h.astype(np.float32) for h in hists)
# H, W      = accum.shape
# gh, gw    = H // 4, W // 4
# grid      = np.array([[accum[r*gh:(r+1)*gh, c*gw:(c+1)*gw].sum()
#                         for c in range(4)] for r in range(4)])
# grid_norm = grid / grid.sum()
# max_cell  = grid_norm.max()
# min_cell  = grid_norm.min()

# uniform_flag = "⚠  BIASED — activity concentrated in one region" if max_cell > 0.15 else "✓  OK"

# print(f"  Cell activity fractions (expected ≈ 0.0625 each):")
# for row in grid_norm:
#     print("    " + "  ".join(f"{v:.4f}" for v in row))
# print(f"    max = {max_cell:.4f}   min = {min_cell:.4f}")
# print(f"    {uniform_flag}")

# # ── Summary ───────────────────────────────────────────────────────────────────
# print("\n── Summary ──────────────────────────────────────────────────────────────")
# print(f"  Sparsity         {flag}")
# print(f"  Temporal spikes  {spike_flag}")
# print(f"  Stagnant pairs   {stagnant_flag}")
# print(f"  Spatial bias     {uniform_flag}")










## Claude overcooking it
# """
# verify_histograms.py
# --------------------
# Standalone verification script for event histogram quality.
# Reads experiment_results.bin (packed binary histograms) and runs a
# suite of checks in increasing complexity.

# Usage:
#     python verify_histograms.py --bin experiment_results.bin \
#                                 --height 480 --width 640 \
#                                 [--baseline_dir outputs/baseline] \
#                                 [--save_plots]

# The binary file is expected to be the hist_records format produced by
# save_exp_video(): each record is np.packbits(binary_hist).tobytes(),
# all records concatenated sequentially with a fixed bytes-per-frame.
# """

# import argparse
# import os
# import sys
# import struct

# import cv2 as cv
# import numpy as np

# # ── optional imports (only needed for higher-level checks) ────────────────────
# try:
#     from scipy.signal import correlate2d
#     SCIPY_AVAILABLE = True
# except ImportError:
#     SCIPY_AVAILABLE = False

# try:
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     import matplotlib.gridspec as gridspec
#     MATPLOTLIB_AVAILABLE = True
# except ImportError:
#     MATPLOTLIB_AVAILABLE = False


# # =============================================================================
# #  LOADING
# # =============================================================================

# def load_histograms(bin_path: str, height: int, width: int) -> list[np.ndarray]:
#     """
#     Unpack all histograms from the .bin file.
#     Returns a list of boolean (H, W) uint8 arrays (0/1).
#     """
#     total_pixels = height * width
#     packed_bytes = int(np.ceil(total_pixels / 8))

#     with open(bin_path, "rb") as f:
#         raw = f.read()

#     if len(raw) % packed_bytes != 0:
#         raise ValueError(
#             f"File size {len(raw)} is not a multiple of expected "
#             f"{packed_bytes} bytes/frame ({height}x{width}). "
#             "Check --height and --width."
#         )

#     n_frames = len(raw) // packed_bytes
#     histograms = []
#     for i in range(n_frames):
#         chunk = raw[i * packed_bytes : (i + 1) * packed_bytes]
#         bits = np.unpackbits(np.frombuffer(chunk, dtype=np.uint8))
#         hist = bits[:total_pixels].reshape(height, width).astype(np.uint8)
#         histograms.append(hist)

#     print(f"[load] Loaded {n_frames} histograms ({height}x{width})")
#     return histograms


# def load_baseline_frames(baseline_dir: str) -> list[np.ndarray]:
#     """Load saved RGB baseline frames, sorted by filename."""
#     if not os.path.isdir(baseline_dir):
#         return []
#     paths = sorted(
#         p for p in (
#             os.path.join(baseline_dir, f) for f in os.listdir(baseline_dir)
#         )
#         if p.lower().endswith((".jpg", ".jpeg", ".png"))
#     )
#     frames = [cv.imread(p) for p in paths]
#     frames = [f for f in frames if f is not None]
#     print(f"[load] Loaded {len(frames)} baseline frames from {baseline_dir}")
#     return frames


# # =============================================================================
# #  LEVEL 1 — SIMPLE SANITY CHECKS (no baseline needed)
# # =============================================================================

# def check_sparsity(histograms: list[np.ndarray]) -> dict:
#     """
#     Active pixel fraction per histogram.
#     Good event histograms are typically sparse (<30% active).
#     A very dense histogram (>60%) usually means noise flooding or a
#     too-low threshold in the emulator.
#     """
#     print("\n── Level 1a: Sparsity ──────────────────────────────────────────────")
#     ratios = [h.mean() for h in histograms]
#     mean_r, std_r = float(np.mean(ratios)), float(np.std(ratios))
#     min_r,  max_r = float(np.min(ratios)),  float(np.max(ratios))

#     flag = "⚠ DENSE" if mean_r > 0.40 else ("⚠ VERY SPARSE" if mean_r < 0.01 else "✓ OK")
#     print(f"  Active pixel fraction  mean={mean_r:.3f}  std={std_r:.3f}  "
#           f"min={min_r:.3f}  max={max_r:.3f}   {flag}")

#     return {"sparsity_mean": mean_r, "sparsity_std": std_r,
#             "sparsity_min": min_r,  "sparsity_max": max_r}


# def check_temporal_consistency(histograms: list[np.ndarray]) -> dict:
#     """
#     Frame-to-frame change in active pixel count.
#     Abrupt spikes → possible dropped frames or accumulator reset artefacts.
#     Near-zero change for many consecutive frames → camera may have stopped.
#     """
#     print("\n── Level 1b: Temporal Consistency ─────────────────────────────────")
#     counts = np.array([h.sum() for h in histograms], dtype=np.float32)
#     deltas = np.abs(np.diff(counts))
#     mean_d, std_d = float(np.mean(deltas)), float(np.std(deltas))
#     spike_thresh = mean_d + 4 * std_d
#     spikes = int(np.sum(deltas > spike_thresh))

#     flag = f"⚠ {spikes} spike(s)" if spikes > 0 else "✓ OK"
#     print(f"  Active pixel Δ/frame   mean={mean_d:.1f}  std={std_d:.1f}   {flag}")

#     # Stagnant windows: <1% of mean count changing
#     stagnant = int(np.sum(deltas < 0.01 * np.mean(counts)))
#     if stagnant:
#         print(f"  ⚠ {stagnant} near-identical consecutive histogram pairs")

#     return {"temporal_mean_delta": mean_d, "temporal_std_delta": std_d,
#             "temporal_spikes": spikes, "stagnant_pairs": stagnant}


# def check_spatial_uniformity(histograms: list[np.ndarray]) -> dict:
#     """
#     Divide the frame into a 4×4 grid. If one quadrant consistently accounts
#     for >50% of all activity, events are spatially biased (clipping, vignetting,
#     or emulator artefact).
#     """
#     print("\n── Level 1c: Spatial Uniformity (4×4 grid) ────────────────────────")
#     accum = sum(h.astype(np.float32) for h in histograms)
#     H, W = accum.shape
#     gh, gw = H // 4, W // 4
#     grid = np.zeros((4, 4), dtype=np.float32)
#     for r in range(4):
#         for c in range(4):
#             grid[r, c] = accum[r*gh:(r+1)*gh, c*gw:(c+1)*gw].sum()

#     grid_norm = grid / grid.sum()
#     # Expected fraction if uniform = 1/16 = 0.0625
#     max_cell = float(grid_norm.max())
#     min_cell = float(grid_norm.min())
#     flag = "⚠ BIASED" if max_cell > 0.15 else "✓ OK"
#     print(f"  Grid cell fraction     max={max_cell:.3f}  min={min_cell:.3f}  "
#           f"(expected ≈0.063)   {flag}")

#     return {"grid_max_fraction": max_cell, "grid_min_fraction": min_cell,
#             "grid_norm": grid_norm.tolist()}


# # =============================================================================
# #  LEVEL 2 — EDGE ALIGNMENT (requires baseline RGB frames)
# # =============================================================================

# def _resize_to_match(src: np.ndarray, target_shape: tuple) -> np.ndarray:
#     H, W = target_shape[:2]
#     if src.shape[:2] != (H, W):
#         src = cv.resize(src, (W, H), interpolation=cv.INTER_NEAREST)
#     return src


# def _rgb_edge_map(frame: np.ndarray, hist_shape: tuple) -> np.ndarray:
#     """Canny edge map of an RGB frame, resized to histogram resolution."""
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
#     gray = _resize_to_match(gray, hist_shape)
#     edges = cv.Canny(cv.GaussianBlur(gray, (3, 3), 0), 30, 90)
#     return (edges > 0).astype(np.uint8)


# def check_edge_alignment(
#     histograms: list[np.ndarray],
#     baseline_frames: list[np.ndarray],
#     dilate_px: int = 3,
# ) -> dict:
#     """
#     Measures IoU between:
#       - the event histogram active pixels
#       - the dilated Canny edge map of the matched (or nearest) baseline frame

#     Dilation compensates for small spatial offsets between the two sources.
#     A score >0.10 is reasonable; >0.20 is good for a sparse binary histogram.
#     """
#     print("\n── Level 2a: Edge Alignment (IoU) ──────────────────────────────────")
#     if not baseline_frames:
#         print("  [SKIP] No baseline frames available.")
#         return {}

#     kernel = cv.getStructuringElement(
#         cv.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
#     )
#     n = min(len(histograms), len(baseline_frames))
#     ious = []
#     recalls = []

#     for i in range(n):
#         hist = histograms[i]
#         edge = _rgb_edge_map(baseline_frames[i], hist.shape)
#         edge_dil = cv.dilate(edge, kernel)

#         intersection = np.logical_and(hist, edge_dil).sum()
#         union        = np.logical_or(hist,  edge_dil).sum()
#         iou  = float(intersection / union)         if union > 0  else 0.0
#         rec  = float(intersection / edge_dil.sum()) if edge_dil.sum() > 0 else 0.0
#         ious.append(iou)
#         recalls.append(rec)

#     mean_iou = float(np.mean(ious))
#     mean_rec = float(np.mean(recalls))
#     flag = "✓ GOOD" if mean_iou > 0.15 else ("⚠ MARGINAL" if mean_iou > 0.06 else "✗ POOR")
#     print(f"  IoU (hist ∩ edges / hist ∪ edges)   mean={mean_iou:.3f}   {flag}")
#     print(f"  Edge recall (how much of edge map is covered)   mean={mean_rec:.3f}")

#     return {"edge_iou_mean": mean_iou, "edge_iou_per_frame": ious,
#             "edge_recall_mean": mean_rec}


# def check_corner_coverage(
#     histograms: list[np.ndarray],
#     baseline_frames: list[np.ndarray],
#     checkerboard_size: tuple = (7, 5),   # inner corners (cols, rows)
#     neighbourhood_px: int = 8,
# ) -> dict:
#     """
#     Detect checkerboard corners in each baseline frame, then check whether
#     those corner locations show activity in the corresponding histogram.

#     Reports: fraction of successfully detected corners that fall within
#     `neighbourhood_px` of an active event pixel.
#     """
#     print("\n── Level 2b: Checkerboard Corner Coverage ──────────────────────────")
#     if not baseline_frames:
#         print("  [SKIP] No baseline frames available.")
#         return {}

#     kernel = np.ones((neighbourhood_px * 2 + 1,) * 2, np.uint8)
#     n = min(len(histograms), len(baseline_frames))

#     coverage_scores = []
#     frames_with_corners = 0

#     for i in range(n):
#         frame = baseline_frames[i]
#         gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
#         found, corners = cv.findChessboardCorners(
#             gray, checkerboard_size,
#             flags=cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
#         )
#         if not found or corners is None:
#             continue
#         frames_with_corners += 1

#         # Refine corners
#         corners = cv.cornerSubPix(
#             gray, corners, (5, 5), (-1, -1),
#             (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
#         )

#         # Map corner coordinates to histogram resolution
#         hist = histograms[i]
#         sy = hist.shape[0] / gray.shape[0]
#         sx = hist.shape[1] / gray.shape[1]

#         # Dilate histogram to create neighbourhood mask
#         hist_dil = cv.dilate(hist, kernel)

#         hits = 0
#         for corner in corners.reshape(-1, 2):
#             cx = int(round(corner[0] * sx))
#             cy = int(round(corner[1] * sy))
#             cx = np.clip(cx, 0, hist.shape[1] - 1)
#             cy = np.clip(cy, 0, hist.shape[0] - 1)
#             if hist_dil[cy, cx]:
#                 hits += 1

#         score = hits / len(corners)
#         coverage_scores.append(score)

#     if not coverage_scores:
#         print("  [SKIP] No checkerboard corners found in any baseline frame.")
#         return {"corner_coverage": None, "frames_with_corners": 0}

#     mean_cov = float(np.mean(coverage_scores))
#     flag = "✓ GOOD" if mean_cov > 0.70 else ("⚠ MARGINAL" if mean_cov > 0.40 else "✗ POOR")
#     print(f"  Corner coverage (fraction of corners near active pixels)")
#     print(f"    mean={mean_cov:.3f}  over {frames_with_corners} frames   {flag}")

#     return {"corner_coverage_mean": mean_cov,
#             "corner_coverage_per_frame": coverage_scores,
#             "frames_with_corners": frames_with_corners}


# # =============================================================================
# #  LEVEL 3 — REPROJECTION CONSISTENCY (requires pose from baseline)
# # =============================================================================

# def check_reprojection_consistency(
#     histograms: list[np.ndarray],
#     baseline_frames: list[np.ndarray],
#     checkerboard_size: tuple = (7, 5),
#     square_size_m: float = 0.025,
#     neighbourhood_px: int = 10,
# ) -> dict:
#     """
#     For each baseline frame where a full checkerboard pose can be solved:
#       1. Estimate camera pose via solvePnP.
#       2. Reproject the 3D checkerboard grid into the histogram frame.
#       3. Measure what fraction of reprojected edge locations are active in
#          the histogram.

#     This ties histogram quality directly to pose-estimation geometry, without
#     needing a neural network. A mean coverage >0.60 is a strong indicator that
#     the histograms encode enough geometric information for pose estimation.
#     """
#     print("\n── Level 3:  Reprojection Consistency ──────────────────────────────")
#     if not baseline_frames:
#         print("  [SKIP] No baseline frames available.")
#         return {}

#     cols, rows = checkerboard_size
#     objp = np.zeros((rows * cols, 3), np.float32)
#     objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m

#     # ── calibrate camera from all frames that have visible corners ────────
#     img_points_all, obj_points_all = [], []
#     frame_shape = None
#     for frame in baseline_frames:
#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
#         found, corners = cv.findChessboardCorners(
#             gray, checkerboard_size,
#             flags=cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
#         )
#         if found:
#             corners = cv.cornerSubPix(
#                 gray, corners, (5, 5), (-1, -1),
#                 (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
#             )
#             img_points_all.append(corners)
#             obj_points_all.append(objp)
#             frame_shape = gray.shape[::-1]  # (W, H)

#     if len(obj_points_all) < 5:
#         print(f"  [SKIP] Only {len(obj_points_all)} frames with corners — "
#               "need ≥5 for calibration.")
#         return {"reprojection_coverage": None}

#     ret, K, dist, _, _ = cv.calibrateCamera(
#         obj_points_all, img_points_all, frame_shape, None, None
#     )
#     print(f"  Camera calibration RMS reprojection error: {ret:.3f} px")

#     # ── generate dense 3D edge points along checkerboard grid lines ───────
#     # Sample points along each row and column of the checkerboard grid
#     edge_pts_3d = []
#     n_interp = 20
#     for r in range(rows):
#         for c in range(cols - 1):
#             for t in np.linspace(0, 1, n_interp):
#                 p = objp[r * cols + c] * (1 - t) + objp[r * cols + c + 1] * t
#                 edge_pts_3d.append(p)
#     for c in range(cols):
#         for r in range(rows - 1):
#             for t in np.linspace(0, 1, n_interp):
#                 p = objp[r * cols + c] * (1 - t) + objp[(r + 1) * cols + c] * t
#                 edge_pts_3d.append(p)
#     edge_pts_3d = np.array(edge_pts_3d, dtype=np.float32)

#     kernel = np.ones((neighbourhood_px * 2 + 1,) * 2, np.uint8)
#     n = min(len(histograms), len(baseline_frames))
#     coverage_scores = []
#     reproj_errors = []

#     for i in range(n):
#         frame = baseline_frames[i]
#         gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
#         found, corners = cv.findChessboardCorners(
#             gray, checkerboard_size,
#             flags=cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
#         )
#         if not found:
#             continue
#         corners = cv.cornerSubPix(
#             gray, corners, (5, 5), (-1, -1),
#             (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
#         )
#         ok, rvec, tvec = cv.solvePnP(objp, corners, K, dist)
#         if not ok:
#             continue

#         # Reprojection error on corners (sanity check)
#         projected_corners, _ = cv.projectPoints(objp, rvec, tvec, K, dist)
#         err = float(np.mean(np.linalg.norm(
#             corners.reshape(-1, 2) - projected_corners.reshape(-1, 2), axis=1
#         )))
#         reproj_errors.append(err)

#         # Project dense edge grid into histogram space
#         proj_edges, _ = cv.projectPoints(edge_pts_3d, rvec, tvec, K, dist)
#         proj_edges = proj_edges.reshape(-1, 2)

#         hist = histograms[i]
#         sy = hist.shape[0] / gray.shape[0]
#         sx = hist.shape[1] / gray.shape[1]
#         hist_dil = cv.dilate(hist, kernel)

#         hits = 0
#         valid = 0
#         for pt in proj_edges:
#             cx = int(round(pt[0] * sx))
#             cy = int(round(pt[1] * sy))
#             if 0 <= cx < hist.shape[1] and 0 <= cy < hist.shape[0]:
#                 valid += 1
#                 if hist_dil[cy, cx]:
#                     hits += 1

#         if valid > 0:
#             coverage_scores.append(hits / valid)

#     if not coverage_scores:
#         print("  [SKIP] Could not solve pose for any frame.")
#         return {"reprojection_coverage": None}

#     mean_cov  = float(np.mean(coverage_scores))
#     mean_rerr = float(np.mean(reproj_errors))
#     flag = "✓ GOOD" if mean_cov > 0.60 else ("⚠ MARGINAL" if mean_cov > 0.35 else "✗ POOR")
#     print(f"  Reprojection coverage (grid edges → histogram activity)")
#     print(f"    mean={mean_cov:.3f}  over {len(coverage_scores)} frames   {flag}")
#     print(f"  Mean corner reprojection error (px): {mean_rerr:.3f}")

#     return {"reprojection_coverage_mean": mean_cov,
#             "reprojection_coverage_per_frame": coverage_scores,
#             "mean_corner_reproj_error_px": mean_rerr}


# # =============================================================================
# #  OPTIONAL: SAVE DIAGNOSTIC PLOTS
# # =============================================================================

# def save_diagnostic_plots(
#     histograms: list[np.ndarray],
#     results: dict,
#     out_dir: str = "outputs/verification",
# ) -> None:
#     if not MATPLOTLIB_AVAILABLE:
#         print("\n[plots] matplotlib not available — skipping plots.")
#         return

#     os.makedirs(out_dir, exist_ok=True)

#     # ── 1. Sparsity over time ─────────────────────────────────────────────
#     if "sparsity_mean" in results:
#         ratios = [h.mean() for h in histograms]
#         fig, ax = plt.subplots(figsize=(10, 3))
#         ax.plot(ratios, color="#2196F3", linewidth=1.2)
#         ax.axhline(np.mean(ratios), color="red", linestyle="--", linewidth=1,
#                    label=f"mean={np.mean(ratios):.3f}")
#         ax.set(xlabel="Histogram index", ylabel="Active pixel fraction",
#                title="Sparsity over time")
#         ax.legend()
#         fig.tight_layout()
#         fig.savefig(os.path.join(out_dir, "sparsity_over_time.png"), dpi=120)
#         plt.close(fig)

#     # ── 2. Accumulated activity heatmap ──────────────────────────────────
#     accum = sum(h.astype(np.float32) for h in histograms)
#     fig, ax = plt.subplots(figsize=(7, 5))
#     im = ax.imshow(accum, cmap="hot", interpolation="nearest")
#     plt.colorbar(im, ax=ax, fraction=0.03)
#     ax.set_title("Accumulated event activity across all histograms")
#     fig.tight_layout()
#     fig.savefig(os.path.join(out_dir, "accumulated_activity.png"), dpi=120)
#     plt.close(fig)

#     # ── 3. Edge IoU per frame (if available) ─────────────────────────────
#     if "edge_iou_per_frame" in results and results["edge_iou_per_frame"]:
#         fig, ax = plt.subplots(figsize=(10, 3))
#         ax.plot(results["edge_iou_per_frame"], color="#4CAF50", linewidth=1.2)
#         ax.axhline(results["edge_iou_mean"], color="red", linestyle="--",
#                    linewidth=1, label=f"mean={results['edge_iou_mean']:.3f}")
#         ax.set(xlabel="Frame index", ylabel="IoU", title="Edge alignment IoU per frame")
#         ax.legend()
#         fig.tight_layout()
#         fig.savefig(os.path.join(out_dir, "edge_iou_per_frame.png"), dpi=120)
#         plt.close(fig)

#     # ── 4. Reprojection coverage per frame (if available) ────────────────
#     if "reprojection_coverage_per_frame" in results and results["reprojection_coverage_per_frame"]:
#         fig, ax = plt.subplots(figsize=(10, 3))
#         ax.plot(results["reprojection_coverage_per_frame"], color="#FF9800", linewidth=1.2)
#         ax.axhline(results["reprojection_coverage_mean"], color="red", linestyle="--",
#                    linewidth=1, label=f"mean={results['reprojection_coverage_mean']:.3f}")
#         ax.set(xlabel="Frame index", ylabel="Coverage",
#                title="Reprojection coverage per frame")
#         ax.legend()
#         fig.tight_layout()
#         fig.savefig(os.path.join(out_dir, "reproj_coverage_per_frame.png"), dpi=120)
#         plt.close(fig)

#     print(f"\n[plots] Saved diagnostic plots to {out_dir}/")


# # =============================================================================
# #  MAIN
# # =============================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description="Verify event histogram quality from experiment_results.bin"
#     )
#     parser.add_argument("--bin",          default="experiment_results.bin",
#                         help="Path to packed histogram binary file")
#     parser.add_argument("--height",       type=int, default=480,
#                         help="Frame height in pixels")
#     parser.add_argument("--width",        type=int, default=640,
#                         help="Frame width in pixels")
#     parser.add_argument("--baseline_dir", default="outputs/baseline",
#                         help="Directory containing baseline RGB JPEG frames")
#     parser.add_argument("--cb_cols",      type=int, default=7,
#                         help="Checkerboard inner corners (columns)")
#     parser.add_argument("--cb_rows",      type=int, default=5,
#                         help="Checkerboard inner corners (rows)")
#     parser.add_argument("--square_mm",    type=float, default=25.0,
#                         help="Checkerboard square size in mm")
#     parser.add_argument("--save_plots",   action="store_true",
#                         help="Save diagnostic plots to outputs/verification/")
#     parser.add_argument("--skip_level",   type=int, default=0, choices=[0, 1, 2, 3],
#                         help="Skip checks above this level (0 = run all)")
#     args = parser.parse_args()

#     if not os.path.isfile(args.bin):
#         print(f"[ERROR] Binary file not found: {args.bin}")
#         sys.exit(1)

#     print("=" * 65)
#     print("  EVENT HISTOGRAM VERIFICATION")
#     print("=" * 65)

#     # Load data
#     histograms      = load_histograms(args.bin, args.height, args.width)
#     baseline_frames = load_baseline_frames(args.baseline_dir)

#     results = {}

#     # ── Level 1 ───────────────────────────────────────────────────────────
#     results.update(check_sparsity(histograms))
#     results.update(check_temporal_consistency(histograms))
#     results.update(check_spatial_uniformity(histograms))

#     if args.skip_level < 2:
#         # ── Level 2 ───────────────────────────────────────────────────────
#         cb = (args.cb_cols, args.cb_rows)
#         results.update(check_edge_alignment(histograms, baseline_frames))
#         results.update(check_corner_coverage(histograms, baseline_frames,
#                                              checkerboard_size=cb))

#     if args.skip_level < 3:
#         # ── Level 3 ───────────────────────────────────────────────────────
#         results.update(check_reprojection_consistency(
#             histograms, baseline_frames,
#             checkerboard_size=(args.cb_cols, args.cb_rows),
#             square_size_m=args.square_mm / 1000.0,
#         ))

#     # ── Summary ───────────────────────────────────────────────────────────
#     print("\n" + "=" * 65)
#     print("  SUMMARY")
#     print("=" * 65)
#     for k, v in results.items():
#         if not isinstance(v, (list, dict)):
#             print(f"  {k:<42} {v}")

#     if args.save_plots:
#         save_diagnostic_plots(histograms, results)

#     print()


# if __name__ == "__main__":
#     main()