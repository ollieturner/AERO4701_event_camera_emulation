
# Setup
# Wait for start/stop recording flag
# Read parameters from file
# Open camera
# Calibrate camera - set/raise flag (keep trying/repeating until successful. if 5 failures, abort)
# Wait for next flag?

# Record video for 30s (stream, + event + spatial?)
# Save frames to file
# Save event data in real time (replicate real)

# Flag?
# Then read through images for baseline pose estimation
# Save to file
# Wipe images 

## LIBRARIES
# Import libraries
import numpy as np
import cv2 as cv
import glob
import os
import shutil
import camera_helper as h

# Logging message
print("Starting camera program...")

## SETUP 
# Setup directories
output_dir, calib_folder = h.setup_directories()

# Calibration parameters 
CHESSBOARD, MAX_BOARDS, SQUARE_SIZE, L, MAX_CALIB_ATTEMPTS = h.setup_calib_parameters()

# Ground truth chessboard corner coordinates
objpoints_3boards = h.get_cboard_gt()


## WAIT FOR START
# TODO threading? check with others for integration


## PREPARE CAMERA
# Read parameters in from file
# TODO Read parameters in from file to open camera?


## OPEN AND CALIBRATE CAMERA 
CALIB_FLAG = False
CALIB_ATTEMPTS = 0

while not CALIB_FLAG and CALIB_ATTEMPTS < MAX_CALIB_ATTEMPTS:
    
    # Take 2s video and save to calibration folder
    # h.save_calib_video()
    
    # Load images
    images = glob.glob(f"{calib_folder}/*.jpeg")
    if len(images) < 5:
        print("Not enough frames, retrying...")
        CALIB_ATTEMPTS += 1
        continue
    
    # Detect chessboards
    objpoints, imgpoints, img_size = h.detect_cboard_calib(images)
    if len(objpoints) < 3:
        print("Not enough valid detections, retrying...")
        CALIB_ATTEMPTS += 1
        continue

    # Calibrate camera
    ret, mtx, dist, _, _ = cv.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # TODO extra: only accept below a max reprojection error (e.g. ret below 2)

    # Logging message
    print("Camera matrix:\n", mtx)

    # Cleanup (remove) calibration folder
    # TODO uncomment this
    # shutil.rmtree(calib_folder, ignore_errors=True)

    CALIB_FLAG = True
    CALIB_ATTEMPTS += 1

print("Camera calibration complete\n")


## SET/WAIT
# TODO check with integration


## RECORD EXPERIMENT 



# Record video for 30s (stream, + event + spatial?)
# Save frames to file
# Save event data in real time (replicate real)