## RUN INSTRUCTIONS
# source venv/bin/activate

# TODO change debug focus from a flag to an lcm message 
# TODO check directories setup 
# TODO update exerpiment output results
# TODO update camera settings file 

# TODO make sure payload comp focuses at mid range
# TODO update chessboard layout to new configuration for right pose estimates 
# TODO put flag if statements inside functions not out here 
# TODO use directory names 

## LIBRARIES
import cv2 as cv
import glob
import camera_helper_new as h
import lcm_helper as lcm_h


## FLAGS
SHOW_CAMERA_FEED    = True   # Display live camera feed widget during capture
SAVE_DEBUG_IMAGES   = False   # Save calibration_test, baseline_pose annotated images and event data
DELETE_IMAGES       = False  # Delete all saved images (calibration + experiment) after each run
DEBUG_MODE         = True   # Save a focused frame to outputs/ after autofocus

# Logging message
print("Starting camera program...")


## OUTER EXPERIMENT LOOP
# Waits for calibration to be enabled, runs calibration + experiment, then loops for the next experiment
while True:

    ## WAIT FOR START
    # Stop program until payload comp msg received and camera is enabled
    start_calib_cam = False
    while start_calib_cam == False:
        msg = lcm_h.wait_for_payload_comp_msg()
        start_calib_cam = msg.cam_enabled


    ## SETUP
    # Setup directories
    output_dir, calib_folder, baseline_folder, baseline_pose_folder = h.setup_directories()

    # Calibration parameters
    CHESSBOARD, MAX_BOARDS, SQUARE_SIZE, L, MAX_CALIB_ATTEMPTS = h.setup_calib_parameters()

    # Ground truth chessboard corner coordinates
    objpoints_3boards = h.get_cboard_gt()

    # Define ROIS of chessboards for calibration
    ROIS = h.define_rois()
    # h.test_draw_rois(image_path="scripts/lcm/pi/test_to_calib_roi.jpg", ROIS=ROIS)


    ## OPEN AND CALIBRATE CAMERA
    CALIB_FLAG = False
    CALIB_ATTEMPTS = 0
    picam2_ = None

    # Attempt calibration until successful or reached max attempts
    while not CALIB_FLAG and CALIB_ATTEMPTS < MAX_CALIB_ATTEMPTS:
        print("Starting calibration...\n")

        # Read parameters in from file
        params = h.prep_pi_cam_params()

        # Open camera and set focus
        picam2_ = h.open_picam(params, picam2_, debug_mode=DEBUG_MODE)
        if picam2_ == None:
            print("Camera opening and focus failed\n")
            CALIB_ATTEMPTS += 1
            continue

        # Take short video and save to calibration folder
        h.save_calib_video_picam(picam2_, SHOW_CAMERA_FEED, calib_time=20.0)

        # Load images
        images = glob.glob(f"outputs/calibration/*.jpeg")
        if len(images) < 5:
            print("Not enough frames, retrying...\n")
            CALIB_ATTEMPTS += 1
            if picam2_ is not None:
                picam2_ = h.close_camera(picam2_)
            continue

        # Detect chessboards
        objpoints, imgpoints, img_size = h.detect_cboard_calib(images, ROIS, save_debug_images=SAVE_DEBUG_IMAGES)
        if len(objpoints) < 3:
            print("Not enough valid detections, retrying...")
            CALIB_ATTEMPTS += 1
            if picam2_ is not None:
                picam2_ = h.close_camera(picam2_)
            continue

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        print("Camera matrix:\n", mtx)

        CALIB_FLAG = True
        CALIB_ATTEMPTS += 1
    
    # Check calibration worked ok
    if CALIB_FLAG == False:
        print("Calibration failed\n")
        if picam2_ is not None:
            picam2_ = h.close_camera(picam2_)
        continue

    # Save calibration settings
    print("Camera calibration complete\n")
    saved_cam_settings = h.extract_applied_settings(picam2_)

    # Cleanly close calibration camera
    picam2_ = h.close_camera(picam2_)


    ## PAYLOAD COMP COMMS
    # Tell payload computer camera calibration status
    lcm_h.publish_cam_msg(cam_calib_complete=CALIB_FLAG)

    # Wait for payload computer confirmation to start experiment
    start_exp = False
    while start_exp == False:
        msg = lcm_h.wait_for_payload_comp_msg()
        start_exp = msg.exp_enabled


    ## RECORD EXPERIMENT
    # Reopen camera with saved settings
    picam2_ = h.open_picam_for_exp(params, picam2_, saved_cam_settings)

    # Check camera opened ok
    if picam2_ is None:
        print("Camera opening and focus failed for experiment\n")
        lcm_h.publish_cam_msg(exp_complete=False)
        continue

    # Run experiment
    exp_success = h.save_exp_video(picam2_, display_widget = SHOW_CAMERA_FEED, save_debug_images = SAVE_DEBUG_IMAGES, exp_time=20.0)


    # Check experiment was ok
    if exp_success == None:
        print("Experiment failed\n")
        lcm_h.publish_cam_msg(exp_complete=False)
        continue


    ## BASELINE DATA PROCESSING
    # Cycle through images in baseline/, estimate poses, save to file
    h.process_baseline_data(objpoints_3boards, mtx, dist, ROIS, save_debug_images=SAVE_DEBUG_IMAGES)


    ## CLEANUP
    h.cleanup_images(DELETE_IMAGES)


    ## FINISH
    # Tell payload computer experiment is finished
    lcm_h.publish_cam_msg(exp_complete=True)

    print("\nExperiment complete. Waiting for next calibration trigger...\n")



#########################
# ~ # Checking reprojection error after calibration
# ~ # h.check_repoj_error(objpoints, rvecs, tvecs, mtx, dist, imgpoints)

