import numpy as np
import cv2 as cv

HEIGHT = 380
WIDTH = 640
NUM_PIXELS = HEIGHT * WIDTH


def load_histogram(file_path):
    # Load packed bitstream
    data = np.load(file_path)

    # Unpack into 0/1 array
    bits = np.unpackbits(data)

    # Trim excess bits (if any padding exists)
    bits = bits[:NUM_PIXELS]

    # Reshape to image
    img = bits.reshape((HEIGHT, WIDTH)).astype(np.uint8)

    return img


file_path = "outputs/experiment_results/histograms/hist_003.npy"

img = load_histogram(file_path)

# Convert to visible image (0 or 255)
vis = img * 255

cv.imshow("Packed Histogram (640x380)", vis)
while True:
    key = cv.waitKey(10) & 0xFF
    if key == 27: # ESC 
        break
      
cv.destroyAllWindows()
