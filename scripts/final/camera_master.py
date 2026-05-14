## RUN INSTRUCTIONS
# .\venv\Scripts\Activate.ps1
# source venv/bin/activate

# TODO test event data saving and running in linux 
# TODO run full fresh test
# TODO check against tests

# TODO check flags/calibration process with Lincoln 
# TODO Prepare raspberry pi camera 3 version


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
output_dir, calib_folder, baseline_folder, baseline_pose_folder = h.setup_directories()

# Calibration parameters 
CHESSBOARD, MAX_BOARDS, SQUARE_SIZE, L, MAX_CALIB_ATTEMPTS = h.setup_calib_parameters()

# Ground truth chessboard corner coordinates
objpoints_3boards = h.get_cboard_gt()

# Define ROIS of chessboards for calibration
ROIS = h.define_rois()
# outputs/IMG_3588.jpeg
h.test_draw_rois(image_path="outputs/calibration/frame_0023.jpeg", ROIS=ROIS)



## WAIT FOR START
# TODO threading? check with others for integration


## PREPARE CAMERA
# Read parameters in from file
args = h.prep_camera_params()


## OPEN AND CALIBRATE CAMERA 
CALIB_FLAG = False
CALIB_ATTEMPTS = 0

while not CALIB_FLAG and CALIB_ATTEMPTS < MAX_CALIB_ATTEMPTS:
    # # Take 2s video and save to calibration folder
    # h.save_calib_video(args)

    # Load images
    images = glob.glob(f"{calib_folder}/*.jpeg")
    if len(images) < 5:
        print("Not enough frames, retrying...")
        CALIB_ATTEMPTS += 1
        continue
    
    # Detect chessboards
    objpoints, imgpoints, img_size = h.detect_cboard_calib(images, ROIS)
    if len(objpoints) < 3:
        print("Not enough valid detections, retrying...")
        CALIB_ATTEMPTS += 1
        continue

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # TODO extra: only accept below a max reprojection error (e.g. ret below 2)

    # Logging message
    print("Camera matrix:\n", mtx)

    # Cleanup (remove) calibration folder
    # TODO uncomment this later
    # shutil.rmtree(calib_folder, ignore_errors=True)

    CALIB_FLAG = True
    CALIB_ATTEMPTS += 1

print("Camera calibration complete\n")

# Checking reprojection error after calibration
# h.check_repoj_error(objpoints, rvecs, tvecs, mtx, dist, imgpoints)



## SET/WAIT
# TODO check with integration


# RECORD EXPERIMENT 
# Take video with baseline images/frames
# TODO add in event camera, histograms and saving the data (for now, save event frames and spatial histograms)
print("Starting experiment")
h.save_exp_video(args)
print("Experiment recording complete")


## BASELINE DATA PROCESSING
print("Processing baseline frames")

# Cycle through images in baseline/, estimate poses, save to file then delete images 
h.process_baseline_data(objpoints_3boards, mtx, dist, ROIS)


