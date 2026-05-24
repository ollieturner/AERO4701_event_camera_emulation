import numpy as np
import cv2 as cv
import glob

# ==============================
# PARAMETERS
# ==============================
CHESSBOARD = (10, 7)   # inner corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (Z=0 plane)
objp = np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

# Storage
objpoints = []
imgpoints = []

images = glob.glob('checkerboard_imgs/*.jpg')

# ==============================
# STEP 1: DETECT CORNERS
# ==============================
print("Detecting checkerboards...")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(
        gray, CHESSBOARD,
        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(
            gray, corners, (11,11), (-1,-1), criteria
        )
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, CHESSBOARD, corners2, ret)
        cv.imshow('Corners', img)
        cv.waitKey(200)

cv.destroyAllWindows()

# ==============================
# STEP 2: CALIBRATION
# ==============================
print("Calibrating camera...")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Camera matrix:\n", mtx)
print("Distortion:\n", dist)

# ==============================
# FUNCTION: DRAW AXES
# ==============================
def draw_axes(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))

    img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 3) # X (blue)
    img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3) # Y (green)
    img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 3) # Z (red)

    return img

# Axis length (in checkerboard units)
axis = np.float32([
    [3,0,0],
    [0,3,0],
    [0,0,-3]
]).reshape(-1,3)

# ==============================
# STEP 3: POSE ESTIMATION
# ==============================
print("Estimating pose...")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, None)

    if ret:
        corners2 = cv.cornerSubPix(
            gray, corners, (11,11), (-1,-1), criteria
        )

        # --- Solve PnP ---
        success, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)

        if success:
            print("\nImage:", fname)
            print("rvec:\n", rvec)
            print("tvec:\n", tvec)

            # Project 3D axes to image
            imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)

            # Draw axes
            img = draw_axes(img, corners2, imgpts)

            cv.imshow('Pose', img)
            cv.waitKey(500)

cv.destroyAllWindows()