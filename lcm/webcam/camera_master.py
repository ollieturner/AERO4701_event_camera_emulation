## RUN INSTRUCTIONS
# .\venv\Scripts\Activate.ps1
# source venv/bin/activate

# TODO test event data saving and running in linux 
# TODO run full fresh test
# TODO check against tests

# TODO check flags/calibration process with Lincoln 
# TODO Prepare raspberry pi camera 3 version

# TODO Add debug mode 
# TODO change fps, process images on go don't save to file - have two different versions
# TODO initialise messages to LCM/variables as false? don't need intialising? 


# TODO makes outputs folder on same level?
# TODO split up message struct?

## LIBRARIES
# Import libraries
import numpy as np
import cv2 as cv
import glob
import os
import shutil
import camera_helper as h
import lcm_helper as lcm_h

# Logging message
print("Starting camera program...")

## SETUP 
# Setup directories
output_dir, calib_folder, baseline_folder, baseline_pose_folder = h.setup_directories()

## TODO: CHANGE TO NEW CHESSBOARD LAYOUT
# Calibration parameters 
CHESSBOARD, MAX_BOARDS, SQUARE_SIZE, L, MAX_CALIB_ATTEMPTS = h.setup_calib_parameters()

## TODO: CHANGE TO NEW CHESSBOARD LAYOUT
# Ground truth chessboard corner coordinates
objpoints_3boards = h.get_cboard_gt()

# Define ROIS of chessboards for calibration
ROIS = h.define_rois()
# h.test_draw_rois(image_path="outputs/calibration/frame_0023.jpeg", ROIS=ROIS)


## WAIT FOR START
# Stop program until payload comp msg received and camera is enabled
# TODO check flow/msg with lincoln
# TODO time limit on number of msg's to check? Do fileno thing to make it time based?
start_calib_cam = False
while start_calib_cam == False:
    msg = lcm_h.wait_for_payload_comp_msg()
    start_calib_cam = msg.cam_enabled


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


# # SET/WAIT
# TODO check with integration
# Tell payload computer camera is calibrated
# TODO change to set flag based on is calibration successful or not
lcm_h.publish_cam_msg(cam_calib_complete = True)

# Wait for payload computer confirmation to start experiment
start_exp = False
while start_exp == False:
    msg = lcm_h.wait_for_payload_comp_msg()
    start_exp = msg.exp_enabled


## RECORD EXPERIMENT 
# Take video with baseline images/frames
print("Starting experiment")
h.save_exp_video(args)
print("Experiment recording complete")


## BASELINE DATA PROCESSING
print("Processing baseline frames")

# Cycle through images in baseline/, estimate poses, save to file then delete images 
h.process_baseline_data(objpoints_3boards, mtx, dist, ROIS)


## FINISH
# Tell payload computer experiment is finished
lcm_h.publish_cam_msg(exp_complete = True)
