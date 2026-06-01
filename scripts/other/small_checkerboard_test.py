import cv2 as cv
import numpy as np

# -----------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------
CHESSBOARD  = (5, 3)        # inner corners (cols, rows)
IMAGE_PATH  = "scripts/image.png"

# -----------------------------------------------------------------------
# LOAD IMAGE
# -----------------------------------------------------------------------
img = cv.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read {IMAGE_PATH}")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
vis  = img.copy()

# -----------------------------------------------------------------------
# DETECT
# -----------------------------------------------------------------------
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, None)

if ret:
    print(f"[INFO] Checkerboard detected — {len(corners)} corners found")
    corners = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    cv.drawChessboardCorners(vis, CHESSBOARD, corners, ret)
else:
    print("[WARNING] Checkerboard not detected")

# -----------------------------------------------------------------------
# DISPLAY
# -----------------------------------------------------------------------
cv.imshow("Detection result", vis)
cv.imwrite("detection_result.png", vis)
cv.waitKey(0)
cv.destroyAllWindows()


# import cv2 as cv

# img = cv.imread("image.png")
# h, w = img.shape[:2]
# print(f"Image size: {w} x {h} pixels")

# physical_square_size_mm = (pixels_per_square / image_width_pixels) * physical_print_width_mm