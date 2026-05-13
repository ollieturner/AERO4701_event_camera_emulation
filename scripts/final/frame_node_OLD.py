# Read from parameters file, open camera
# Checkerboard calibration
# Stream camera + checkerboard detection (mask out sequentially)
# Save pose estimates to file 

# Save videos from run, then do pose estimates - is too slow to do it on the fly? (check without having to draw on pictures)
# Then clear/delete photos
# Ways to make faster - resize image, reduce CornerSubPix, only calibrate camera once, save then load them in
# Try findChessboardCornersSB

## LIBRARIES
# Import libraries
import numpy as np
import cv2 as cv
import glob
import os

import scripts.final.frame_helper_funcs_OLD as h

# Logging message
print("Starting baseline camera program...")

## BOARD PARAMETERS
CHESSBOARD = (5, 3)
MAX_BOARDS = 3
SQUARE_SIZE = 0.00225       # in metres

L = 0.034511                # for triangular positions of boards
h = np.sqrt(3)/2 * L

board_centres = np.array([
    [0,  2*h/3, 0],        # top
    [-L/2, -h/3, 0],       # bottom-left
    [ L/2, -h/3, 0]        # bottom-right
])


## SETUP 
# Create output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Define ground truth corner coordinates for 3 boards 
objpoints_3boards = []
grid = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
grid = np.hstack([grid * SQUARE_SIZE, np.zeros((grid.shape[0], 1))])

for centre in board_centres:
    R = h.rotation_towards(centre)

    rotated = (R @ grid.T).T
    translated = rotated + centre

    objpoints_3boards.append(translated.astype(np.float32))


## OPEN CAMERA 
# Read parameters in from file 

# (load image)
images = glob.glob('scripts/pose_estimation/board_imgs/*.jpeg')


# TODO record video for 2s? Save frames to file (then wipe after calibration)



## CAMERA CALIBRATION
print("Calibrating camera...")
objpoints = []
imgpoints = []
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Calibrate camera based on first checkerboard detected
# TODO make program wait until calibrated then proceed (set flag?_
# TODO define region of interest to ensure you get the same checkerboard each time
objpoints, imgpoints, img_size = h.detect_chessboards_calib(images, objpoints, imgpoints, criteria)

# Calibrate camera intrinsics
ret, mtx, dist, _, _ = cv.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

# Logging message
print("Camera matrix:\n", mtx)


## SAVE VIDEO
# TODO run for timer 




axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]]).reshape(-1,3)


