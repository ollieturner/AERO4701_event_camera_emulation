import numpy as np
import cv2 as cv
import time
import shutil
import os
import argparse
import sys
import glob

from picamera2 import Picamera2
from libcamera import controls
from event_camera_emulation.emulator import EventCameraEmulator


def setup_directories():
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    calib_folder = "outputs/calibration"
    os.makedirs(calib_folder, exist_ok=True)
    baseline_folder = "outputs/baseline"
    os.makedirs(baseline_folder, exist_ok=True)
    baseline_pose_folder = "outputs/baseline_pose"
    os.makedirs(baseline_pose_folder, exist_ok=True)

    return output_dir, calib_folder, baseline_folder, baseline_pose_folder


def setup_calib_parameters():
    CHESSBOARD = (5, 3)
    MAX_BOARDS = 3
    SQUARE_SIZE = 0.00225       # in metres
    L = 0.034511                # for triangular positions of boards
    MAX_CALIB_ATTEMPTS = 3
    
    return CHESSBOARD, MAX_BOARDS, SQUARE_SIZE, L, MAX_CALIB_ATTEMPTS

# def define_rois():
#     ROIS = [
#         (0, 0, 640, 360),     # board 1
#         (640, 0, 1280, 360),  # board 2
#         (320, 360, 960, 720)  # board 3
#     ]

#     return ROIS

# Define ROIS centred in the image
# Scale controls ROI size relative to image size
def define_rois(width=640, height=480, scale=0.38):
    w = int(width * scale)
    h = int(height * scale)

    cx = width // 2
    cy = height // 2

    # offsets for triangular layout
    dx = int(width * 0.16)
    dy = int(height * 0.12)

    top_extra = int(height * 0.1)   # extra upward movement

    ROIS = [
        # top board
        (cx - w//2, cy - h//2 - dy - top_extra,
         cx + w//2, cy + h//2 - dy - top_extra),

        # bottom-left board
        (cx - w//2 - dx, cy - h//2 + dy,
         cx + w//2 - dx, cy + h//2 + dy),

        # bottom-right board
        (cx - w//2 + dx, cy - h//2 + dy,
         cx + w//2 + dx, cy + h//2 + dy),
    ]

    return ROIS

# Test function to draw on ROIS onto image for validation
# TODO change to open camera
def test_draw_rois(image_path, ROIS, output_dir="outputs", name="roi_debug.png"): 

    img = cv.imread(image_path)

    # print(img.shape[::-1])

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    vis = img.copy()

    for i, roi in enumerate(ROIS):
        x1, y1, x2, y2 = roi

        # draw rectangle
        cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # label
        cv.putText(vis,f"Board {i}", (x1 + 10, y1 + 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    out_path = os.path.join(output_dir, name)
    cv.imwrite(out_path, vis)

    print(f"Saved ROI test visualisation to: {out_path}")


def rotation_towards(origin):
    direction = origin / np.linalg.norm(origin)
    z = np.array([0, 0, 1])

    # crude but effective: align board normal with outward vector
    v = np.cross(z, direction)
    s = np.linalg.norm(v)
    c = np.dot(z, direction)

    if s == 0:
        return np.eye(3)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    return R


def get_cboard_gt(L = 0.034511, CHESSBOARD = (5, 3), SQUARE_SIZE = 0.00225):
    # Define 3 chessboard corner coordinates 
    h = np.sqrt(3)/2 * L

    board_centres = np.array([
        [0,  2*h/3, 0],        # top
        [-L/2, -h/3, 0],       # bottom-left
        [ L/2, -h/3, 0]        # bottom-right
    ])

    # Define ground truth corner coordinates for 3 boards 
    objpoints_3boards = []
    grid = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
    grid = np.hstack([grid * SQUARE_SIZE, np.zeros((grid.shape[0], 1))])

    for centre in board_centres:
        R = rotation_towards(centre)

        rotated = (R @ grid.T).T
        translated = rotated + centre

        objpoints_3boards.append(translated.astype(np.float32))

    return objpoints_3boards


def detect_cboard_calib(images, ROIS, CHESSBOARD=(5,3), SQUARE_SIZE=0.00225):

    debug_folder="outputs/calibration_test"

    # Create debug folder
    shutil.rmtree(debug_folder, ignore_errors=True)
    os.makedirs(debug_folder, exist_ok=True)

    print("Calibrating camera...")

    if ROIS is None:
        raise ValueError("ROIs must be defined for multi-board calibration")

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    num_boards = len(ROIS)

    # Structured storage
    objpoints = [[] for _ in range(num_boards)]
    imgpoints = [[] for _ in range(num_boards)]

    # Precompute object model once (same for all boards)
    objp_base = np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
    objp_base[:, :2] = np.mgrid[0:CHESSBOARD[0],
                                0:CHESSBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    save_idx = 0
    
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        debug_img = img.copy()

        for board_id, roi in enumerate(ROIS):

            x1, y1, x2, y2 = roi

            # Mask out outside ROI
            mask = np.zeros_like(gray, dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

            # apply mask (keep full image size!)
            masked_img = cv.bitwise_and(gray, gray, mask=mask)

            # TODO could add in exhaustive for calibration? experiment
            ret, corners = cv.findChessboardCorners(masked_img, CHESSBOARD, None)
            # ret, corners = cv.findChessboardCornersSB(masked_img, CHESSBOARD, 
            #         flags = cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_ACCURACY)

            if not ret:
                continue

            corners2 = cv.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)

            objpoints[board_id].append(objp_base.copy())
            imgpoints[board_id].append(corners)

            # Draw corners onto debug image
            cv.drawChessboardCorners(
                debug_img,
                CHESSBOARD,
                corners2,
                ret
            )

            # Draw ROI rectangle
            cv.rectangle(
                debug_img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        # Save annotated image
        save_path = f"{debug_folder}/calib_{save_idx:04d}.png"
        cv.imwrite(save_path, debug_img)

        save_idx += 1

    img_size = gray.shape[::-1]

    # Flatten
    obj_flat = []
    img_flat = []

    for b in range(len(objpoints)):
        for i in range(len(objpoints[b])):
            obj_flat.append(objpoints[b][i])
            img_flat.append(imgpoints[b][i])

    return obj_flat, img_flat, img_size


def check_repoj_error(objpoints, rvecs, tvecs, mtx, dist, imgpoints):
    
    # Calculate mean reprojection error

    total_error = 0

    for i in range(len(objpoints)):

        # Project 3D object points back into image
        imgpoints2, _ = cv.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            mtx,
            dist
        )

        # Compute L2 reprojection error
        error = cv.norm(
            imgpoints[i],
            imgpoints2,
            cv.NORM_L2
        ) / len(imgpoints2)

        total_error += error

    mean_error = total_error / len(objpoints)

    print(f"Mean reprojection error: {mean_error:.6f} pixels")


# No ROI consideration
# def detect_cboard_calib(images, CHESSBOARD = (5, 3), SQUARE_SIZE = 0.00225):
#     print("Calibrating camera...")

#     objpoints = []
#     imgpoints = []
#     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
#     for fname in images:
#         img = cv.imread(fname)
#         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#         # TODO replace with findChessboardCornersSB
#         ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, None)

#         if ret:
#             objp_calib = np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
#             objp_calib[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE
#             objpoints.append(objp_calib)
#             # TODO Reduce from 11, 11?
#             corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
#             imgpoints.append(corners2)
        
#     img_size = gray.shape[::-1]

#     return objpoints, imgpoints, img_size

def prep_webcam_params(camera_file = "camera_settings/camera_settings.txt"):
    params = {}

    with open(camera_file, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            key, value = line.split("=")
            params[key.strip()] = value.strip()

    # Convert numeric fields if needed
    params["video_device"] = int(params.get("video_device", 0))

    return params


def prep_pi_cam_params(camera_file="camera_settings/pi_camera_settings.txt"):
    params = {}

    with open(camera_file, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)
            params[key.strip()] = value.strip()

    # Convert numeric fields
    params["width"]       = int(params.get("width", 640))
    params["height"]      = int(params.get("height", 480))
    params["fps"]         = float(params.get("fps", 50.0))
    params["threshold"]   = float(params.get("threshold", 30.0))
    params["warmup"]      = float(params.get("warmup", 1.0))
    params["grayscale"]   = params.get("grayscale", "false").lower() == "true"

    # Optional overrides - remain None if not set in file
    exposure_us = params.get("exposure_us", "").strip()
    params["exposure_us"] = int(exposure_us) if exposure_us else None

    gain = params.get("gain", "").strip()
    params["gain"] = float(gain) if gain else None

    return params


def get_valid_frame(camera):
    frame = camera.capture_array("main")
    if frame is None or frame.size == 0:
        return None

    # Ensure 3-channel BGR
    if len(frame.shape) != 3:
        return None

    if frame.shape[2] == 4:
        frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
    elif frame.shape[2] != 3:
        return None

    return frame


# Open pi camera 3 and set autofocus 
def open_picam(params, picam2_):
    frame_us = int(1_000_000 / params["fps"])

    try:
        picam2_ = Picamera2()

        config = picam2_.create_video_configuration(
            main={"format": "BGR888", "size": (params["width"], params["height"])},
            controls={"FrameDurationLimits": (frame_us, frame_us)}
        )
        picam2_.configure(config)
        picam2_.start()

        # Let auto algorithms settle before focusing
        time.sleep(params["warmup"])

        # Try autofocus once, then lock the focus position
        try:
            picam2_.set_controls({"AfMode": controls.AfModeEnum.Auto})
            success = picam2_.autofocus_cycle()
            meta = picam2_.capture_metadata()
            lens_pos = meta.get("LensPosition", None)
            if success and lens_pos is not None:
                picam2_.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": lens_pos
                })
        except Exception:
            print(f'[open_picam_save_calib] [ERROR] Could not focus Raspberry Pi camera: {exc}')
            sys.exit(1)

        # Read settled metadata and lock camera state
        meta = picam2_.capture_metadata()
        settled_exposure = int(meta.get("ExposureTime", min(5000, frame_us // 2)))
        settled_gain = float(meta.get("AnalogueGain", 1.0))
        colour_gains = meta.get("ColourGains", None)

        # Use exposure and gain from camera focus, or override if manual provided
        if params["exposure_us"] is not None:
            settled_exposure = params["exposure_us"]
        else:
            settled_exposure = min(settled_exposure, max(1000, int(0.6 * frame_us)))

        if params["gain"] is not None:
            settled_gain = params["gain"]

        lock_controls = {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": settled_exposure,
            "AnalogueGain": settled_gain,
            "FrameDurationLimits": (frame_us, frame_us),
        }

        if colour_gains is not None:
            lock_controls["ColourGains"] = colour_gains

        picam2_.set_controls(lock_controls)

        # Give time to settle
        time.sleep(0.2)

        # Test frame from camera
        previous_image = get_valid_frame(picam2_)
        if previous_image is None:
            print('[stream_camera_events] [ERROR] No valid initial frame returned.')
            sys.exit(1)

        if params["grayscale"]:
            previous_image = cv.cvtColor(previous_image, cv.COLOR_BGR2GRAY)
            previous_image = cv.cvtColor(previous_image, cv.COLOR_GRAY2BGR)

        # Log status
        print(f'[open_picam_save_calib] [INFO] FPS: {params["fps"]:.1f}, frame time: {frame_us} us')
        print(f'[open_picam_save_calib] [INFO] Locked exposure: {settled_exposure} us')
        print(f'[open_picam_save_calib] [INFO] Locked gain: {settled_gain:.3f}')

        return picam2_

    except Exception as exc:
        print(f'[open_picam_save_calib] [ERROR] Could not access Raspberry Pi camera: {exc}')
        sys.exit(1)


# Save calibration video with pi camera 3
def save_calib_video_picam(picam2_, calib_time=1.0, calib_folder="outputs/calibration"):
    # Capture calibration frames
    frames = []
    start_time = time.time()
    while time.time() - start_time < calib_time:
        frame = picam2_.capture_array("main")
        if frame is None:
            continue
        frames.append(frame.copy())

    # TODO remove this?
    # Make calibration folder
    shutil.rmtree(calib_folder, ignore_errors=True)
    os.makedirs(calib_folder, exist_ok=True)

    # Save frames to calibration folder
    for i, frame in enumerate(frames):
        # if params.get("grayscale"):
        #     frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #     frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        cv.imwrite(f"{calib_folder}/frame_{i:04d}.jpeg", frame)

    print(f'[open_picam_save_calib] [INFO] Saved {len(frames)} calibration frames to {calib_folder}')

    return picam2_



def save_exp_video(picam2_, exp_time=5.0, WINDOW_SIZE = 10, baseline_folder="outputs/baseline"):

    # Setup output folders
    # TODO check if needed here?
    shutil.rmtree(baseline_folder, ignore_errors=True)
    os.makedirs(baseline_folder, exist_ok=True)
    event_frame_dir = "event_data/frames"
    event_hist_dir = "event_data/histograms"
    shutil.rmtree("event_data", ignore_errors=True)
    os.makedirs(event_frame_dir, exist_ok=True)
    os.makedirs(event_hist_dir, exist_ok=True)

    # Initialise event camera emulator
    e_camera_emulator = EventCameraEmulator()

    frames = []
    frame_idx = 0
    start_time = time.time()

    # Read first frame
    frame = picam2_.capture_array("main")
    if frame is None:
        print("[save_exp_video] [INFO] Could not read first frame")
        sys.exit()
    prev_frame = frame

    event_hist = None

    # Record data for experiment time
    while time.time() - start_time < exp_time:
        # Read frame
        frame = picam2_.capture_array("main")
        if frame is None:
            continue
        
        # Add frame
        frames.append(frame)

        # Emulate event camera
        event_image = e_camera_emulator.get_events_image_rgb(
            frame,
            prev_frame,
            30,
            record_off_events=True,
            register_off_events_as_on=False
        )

        visual_event_image = e_camera_emulator.get_visual_events_image(event_image)
        prev_frame = frame

        # Save event frame (TODO for debugging)
        cv.imwrite(
            os.path.join(event_frame_dir, f"event_{frame_idx:04d}.png"),
            visual_event_image
        )

        # Add to spatial histogram
        if event_image.ndim == 3:
            gray_event = cv.cvtColor(event_image, cv.COLOR_BGR2GRAY)
        else:
            gray_event = event_image

        gray_event = gray_event.astype(np.float32)

        if event_hist is None:
            event_hist = np.zeros_like(gray_event, dtype=np.float32)

        event_hist += np.abs(gray_event)

        # Save event spatial histogram after accumulation
        if frame_idx % WINDOW_SIZE == 0 and frame_idx > 0:
            # Save raw histogram data
            raw_path = os.path.join(event_hist_dir, f"event_hist_raw_{frame_idx:05d}.npy")
            np.save(raw_path, event_hist)

            # Save normalised visualisation of histogram data
            hist_vis = cv.normalize(event_hist, None, 0, 255, cv.NORM_MINMAX)
            hist_vis = hist_vis.astype(np.uint8)
            vis_path = os.path.join(event_hist_dir, f"event_hist_vis_{frame_idx:05d}.png")
            cv.imwrite(vis_path, hist_vis)

            # Reset window
            event_hist = np.zeros_like(gray_event, dtype=np.float32)

        frame_idx += 1

    if picam2_ is not None:
        picam2_.stop()

    # Save RGB frames for baseline
    for i, frame in enumerate(frames):
        cv.imwrite(f"{baseline_folder}/frame_{i:04d}.jpeg", frame)
    








###############################################



# TODO could make timing better
def save_calib_video_webcam(args, calib_time = 1.0, calib_folder = "outputs/calibration"):
    # Take a 2 second video and save frames - use VideoCapture/imwrite for high quality
    # try:
    #     camera_device = cv.VideoCapture(int(args.video_device))
    # except ValueError:
    #     camera_device = cv.VideoCapture(args.video_device)

    try:
        camera_device = cv.VideoCapture(int(args["video_device"]))
    except ValueError:
        camera_device = cv.VideoCapture(args["video_device"])

    if not camera_device.isOpened():
        print('Could not access camera')
        sys.exit()

    frames = []
    start_time = time.time()

    while time.time() - start_time < calib_time:
        ret, frame = camera_device.read()
        if not ret:
            continue

        frames.append(frame)

    camera_device.release()

    # Save video to calibration folder TODO check this
    shutil.rmtree(calib_folder, ignore_errors=True)
    os.makedirs(calib_folder, exist_ok=True)

    for i, frame in enumerate(frames):
        cv.imwrite(f"{calib_folder}/frame_{i:04d}.jpeg", frame)


def save_exp_video_webcam(args, exp_time=5.0, WINDOW_SIZE = 10, baseline_folder="outputs/baseline"):

    import cv2 as cv
    import numpy as np
    import os
    import time
    import shutil

    from event_camera_emulation.emulator import EventCameraEmulator

    # Setup camera
    try:
        camera_device = cv.VideoCapture(int(args["video_device"])) # args.video_device))
    except ValueError:
        camera_device = cv.VideoCapture(args["video_device"]) # args.video_device)

    if not camera_device.isOpened():
        print("Could not access camera")
        sys.exit()

    # Setup output folders
    shutil.rmtree(baseline_folder, ignore_errors=True)
    os.makedirs(baseline_folder, exist_ok=True)
    event_frame_dir = "event_data/frames"
    event_hist_dir = "event_data/histograms"
    shutil.rmtree("event_data", ignore_errors=True)
    os.makedirs(event_frame_dir, exist_ok=True)
    os.makedirs(event_hist_dir, exist_ok=True)

    # Initialise event camera emulator
    e_camera_emulator = EventCameraEmulator()

    frames = []
    frame_idx = 0
    start_time = time.time()

    # Check first frame
    ret, prev_frame = camera_device.read()
    if not ret:
        print("Could not read first frame")
        sys.exit()

    event_hist = None

    # Record data for experiment time
    while time.time() - start_time < exp_time:
        # Read frame
        ret, frame = camera_device.read()
        if not ret:
            continue
        
        # Add frame
        frames.append(frame)

        # Emulate event camera
        event_image = e_camera_emulator.get_events_image_rgb(
            frame,
            prev_frame,
            30,
            record_off_events=True,
            register_off_events_as_on=False
        )

        visual_event_image = e_camera_emulator.get_visual_events_image(event_image)
        prev_frame = frame

        # Save event frame (TODO for debugging)
        cv.imwrite(
            os.path.join(event_frame_dir, f"event_{frame_idx:04d}.png"),
            visual_event_image
        )

        # Add to spatial histogram
        if event_image.ndim == 3:
            gray_event = cv.cvtColor(event_image, cv.COLOR_BGR2GRAY)
        else:
            gray_event = event_image

        gray_event = gray_event.astype(np.float32)

        if event_hist is None:
            event_hist = np.zeros_like(gray_event, dtype=np.float32)

        event_hist += np.abs(gray_event)

        # Save event spatial histogram after accumulation
        if frame_idx % WINDOW_SIZE == 0 and frame_idx > 0:
            # Save raw histogram data
            raw_path = os.path.join(event_hist_dir, f"event_hist_raw_{frame_idx:05d}.npy")
            np.save(raw_path, event_hist)

            # Save normalised visualisation of histogram data
            hist_vis = cv.normalize(event_hist, None, 0, 255, cv.NORM_MINMAX)
            hist_vis = hist_vis.astype(np.uint8)
            vis_path = os.path.join(event_hist_dir, f"event_hist_vis_{frame_idx:05d}.png")
            cv.imwrite(vis_path, hist_vis)

            # Reset window
            event_hist = np.zeros_like(gray_event, dtype=np.float32)

        frame_idx += 1

    camera_device.release()

    # Save RGB frames for baseline
    for i, frame in enumerate(frames):
        cv.imwrite(f"{baseline_folder}/frame_{i:04d}.jpeg", frame)










# Cycling through frames, estimate pose then save to file 
def process_baseline_data(objpoints_3boards, mtx, dist, ROIS, CHESSBOARD = (5, 3), baseline_folder = "outputs/baseline", pose_folder = "outputs/baseline_pose"):
    # Read in images 
    # images = glob.glob(f"{baseline_folder}/*.jpeg")
    images = sorted(
        glob.glob(os.path.join(baseline_folder, "*.jpeg"))
    )
    print("Found images:", len(images))

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
   
    # Define pose estimate file
    pose_txt_path = os.path.join(pose_folder, "baseline_poses.txt")
    with open(pose_txt_path, "w") as pose_file:

        for fname in images:
            # Extract image
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Extract naming information
            base = os.path.basename(fname)
            name, _ = os.path.splitext(base)
            # TODO Can delete this? leave for clarity now
            pose_file.write(f"Image: {name}\n")

            for board_id, roi in enumerate(ROIS):

                x1, y1, x2, y2 = roi

                # Mask out outside ROI
                mask = np.zeros_like(gray, dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255

                # Apply mask
                working_img = cv.bitwise_and(gray, gray, mask=mask)

                # Detect chessboard corners
                # ret, corners = cv.findChessboardCornersSB(working_img, CHESSBOARD, 
                #     flags = cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_ACCURACY)
                ret, corners = cv.findChessboardCorners(working_img, CHESSBOARD, None)

                # Skip if no image
                if not ret:
                    # Blank out pose estimate to preserve order
                    pose_file.write(f"Board {board_id+1}\n")
                    pose_file.write("rvec: 0 0 0\n")
                    pose_file.write("tvec: 0 0 0\n\n")
                    continue

                corners = cv.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)

                # Pose estimation
                obj_model = objpoints_3boards[board_id]
                success, rvec, tvec = cv.solvePnP(obj_model, corners, mtx, dist)

                # Write out pose estimate
                if success:
                    pose_file.write(f"Board {board_id+1}\n")
                    pose_file.write(
                        f"rvec: {rvec[0][0]} {rvec[1][0]} {rvec[2][0]}\n"
                    )
                    pose_file.write(
                        f"tvec: {tvec[0][0]} {tvec[1][0]} {tvec[2][0]}\n\n"
                    )
                else:
                    pose_file.write(f"Board {board_id+1}\n")
                    pose_file.write("rvec: 0 0 0\n")
                    pose_file.write("tvec: 0 0 0\n\n")

                # TODO Comment out below later
                # Drawing board onto image (debugging)
                if success:  
                    # imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)

                    # Draw corners with thicker visualisation
                    cv.drawChessboardCorners(img, CHESSBOARD, corners, False)

                    corners_int = corners.astype(int)

                    for i in range(len(corners_int) - 1):
                        cv.line(img, tuple(corners_int[i][0]), tuple(corners_int[i+1][0]), (0, 255, 255), 5)

                    # Label board ID
                    label = board_id + 1
                    corner = tuple(corners[0].ravel().astype(int))

                    cv.putText(img, f"{label}", (corner[0] + 10, corner[1] - 30), cv.FONT_HERSHEY_SIMPLEX,
                        5, (0, 0, 255), 6, cv.LINE_AA)
            
            # Save image
            base = os.path.basename(fname)
            name, _ = os.path.splitext(base)
            output_name = os.path.join(pose_folder, f"{name}_multi_pose.png")
            success = cv.imwrite(output_name, img)
            # print(f"SAVE {output_name} -> {success}")
    
    # # Delete contents of baseline folder
    # for file_path in glob.glob(os.path.join(baseline_folder, "*")):
    #     try:
    #         os.remove(file_path)
    #         print(f"Deleted: {file_path}")
    #     except Exception as e:
    #         print(f"Failed to delete {file_path}: {e}")


