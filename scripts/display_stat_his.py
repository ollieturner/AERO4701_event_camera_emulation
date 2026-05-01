import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("event_histograms/hist_vis_00050.png", cv2.IMREAD_GRAYSCALE)

values = img.ravel()

plt.hist(values, bins=100)
plt.title("Event Activity Distribution (0.2s window)")
plt.xlabel("Event intensity")
plt.ylabel("Number of pixels")
plt.show()