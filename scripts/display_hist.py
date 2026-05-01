#!/usr/bin/env python3

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# source hist_venv/bin/activate
# python display_hist.py

# ==============================
# FOLDERS
# ==============================
HIST_DIR = "event_histograms"

# collect files
img_files = sorted([f for f in os.listdir(HIST_DIR) if f.startswith("hist_vis_")])
acc_files = sorted([f for f in os.listdir(HIST_DIR) if f.startswith("event_hist_")])
val_files = sorted([f for f in os.listdir(HIST_DIR) if f.startswith("value_hist_")])

frame_idx = 0

# ==============================
# DISPLAY LOOP
# ==============================
while True:

    # ------------------------------
    # LOAD SPATIAL HISTOGRAM IMAGE
    # ------------------------------
    img_path = os.path.join(HIST_DIR, img_files[frame_idx])
    hist_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # ------------------------------
    # LOAD ACCUMULATED HISTOGRAM
    # ------------------------------
    acc_path = os.path.join(HIST_DIR, acc_files[frame_idx])
    event_hist = np.load(acc_path)

    acc_vis = cv2.normalize(event_hist, None, 0, 255, cv2.NORM_MINMAX)
    acc_vis = acc_vis.astype(np.uint8)

    # ------------------------------
    # LOAD VALUE HISTOGRAM
    # ------------------------------
    val_path = os.path.join(HIST_DIR, val_files[frame_idx])
    value_hist = np.load(val_path)

    # ==============================
    # PLOT VALUE HISTOGRAM (matplotlib)
    # ==============================
    plt.clf()
    plt.title("Event Intensity Distribution")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.plot(value_hist)
    plt.pause(0.001)

    # ==============================
    # SHOW IMAGES
    # ==============================
    cv2.imshow("Spatial Event Heatmap", hist_img)
    cv2.imshow("Accumulated Event Histogram", acc_vis)

    key = cv2.waitKey(0) & 0xFF

    # ------------------------------
    # CONTROLS
    # ------------------------------
    if key == ord('q'):
        break
    elif key == ord('d'):  # next frame
        frame_idx = min(frame_idx + 1, len(img_files) - 1)
    elif key == ord('a'):  # previous frame
        frame_idx = max(frame_idx - 1, 0)

cv2.destroyAllWindows()