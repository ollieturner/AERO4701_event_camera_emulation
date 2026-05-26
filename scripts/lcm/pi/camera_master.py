## RUN INSTRUCTIONS
# source venv/bin/activate

# notes
# TODO might need to make it focus at mid range not closest point?


## LIBRARIES
# Import libraries
import cv2 as cv
import glob
import scripts.lcm.pi.camera_helper as h
import scripts.lcm.pi.lcm_helper as lcm_h


# Logging message
print("Starting camera program...")

## WAIT FOR START
# Stop program until payload comp msg received and camera is enabled
start_calib_cam = False
while start_calib_cam == False:
    msg = lcm_h.wait_for_payload_comp_msg()
    start_calib_cam = msg.cam_enabled

## SETUP 
# Setup directories
# output_dir, calib_folder, baseline_folder, baseline_pose_folder = h.setup_directories()


## TODO: CHANGE TO NEW CHESSBOARD LAYOUT
# Calibration parameters 
CHESSBOARD, MAX_BOARDS, SQUARE_SIZE, L, MAX_CALIB_ATTEMPTS = h.setup_calib_parameters()

## TODO: CHANGE TO NEW CHESSBOARD LAYOUT
# Ground truth chessboard corner coordinates
objpoints_3boards = h.get_cboard_gt()

# Define ROIS of chessboards for calibration
ROIS = h.define_rois()
# h.test_draw_rois(image_path="scripts/pi/test_to_calib_roi.jpg", ROIS=ROIS)


## OPEN AND CALIBRATE CAMERA 
CALIB_FLAG = False
CALIB_ATTEMPTS = 0
picam2_ = None

while not CALIB_FLAG and CALIB_ATTEMPTS < MAX_CALIB_ATTEMPTS:
    print("Starting calibration...\n")
    # Read parameters in from file
    params = h.prep_pi_cam_params()
    
    # Open camera and set focus
    picam2_ = h.open_picam(params, picam2_)

    # Take short video and save to calibration folder
    # h.save_calib_video_picam(picam2_)
    h.save_calib_video_picam_widget(picam2_, calib_time = 20.0)
    
    # Load images
    images = glob.glob(f"outputs/calibration/*.jpeg")
    if len(images) < 5:
        print("Not enough frames, retrying...\n")
        CALIB_ATTEMPTS += 1
        if picam2_ is not None:
            picam2_.stop()
            picam2_.close()
            picam2_ = None
        continue
    
    # Detect chessboards
    objpoints, imgpoints, img_size = h.detect_cboard_calib(images, ROIS)
    if len(objpoints) < 3:
        print("Not enough valid detections, retrying...")
        CALIB_ATTEMPTS += 1
        if picam2_ is not None:
            picam2_.stop()
            picam2_.close()
            picam2_ = None
        continue

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    # Logging message
    print("Camera matrix:\n", mtx)

    # Cleanup (remove) calibration folder
    # shutil.rmtree(calib_folder, ignore_errors=True)

    CALIB_FLAG = True
    CALIB_ATTEMPTS += 1

print("Camera calibration complete\n")
saved_cam_settings = h.extract_applied_settings(picam2_)
print("[INFO] Saved calibration camera settings:", saved_cam_settings)
print("\n")

# Cleanly close calibration camera
picam2_ = h.close_camera(picam2_)


# # PAYLOAD COMP COMMS
# Tell payload computer camera calibration status
# time.sleep(2)
lcm_h.publish_cam_msg(cam_calib_complete = CALIB_FLAG)

# Wait for payload computer confirmation to start experiment
start_exp = False
while start_exp == False:
    msg = lcm_h.wait_for_payload_comp_msg()
    start_exp = msg.exp_enabled


# RECORD EXPERIMENT 
# Take video with baseline images/frames
# Reopen camera with saved settings
picam2_ = h.open_picam_for_exp(params, picam2_, saved_cam_settings)
    
# h.save_exp_video(picam2_)
h.save_exp_video_widget(picam2_)


## BASELINE DATA PROCESSING
# Cycle through images in baseline/, estimate poses, save to file then delete images 
h.process_baseline_data(objpoints_3boards, mtx, dist, ROIS)


## FINISH
# Tell payload computer experiment is finished
lcm_h.publish_cam_msg(exp_complete = True)



#########################
# ~ # Checking reprojection error after calibration
# ~ # h.check_repoj_error(objpoints, rvecs, tvecs, mtx, dist, imgpoints)





