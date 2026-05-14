import numpy as np
import cv2 as cv
import time
import shutil
import os
import argparse
import sys
import glob

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
    dy = int(height * 0.16)

    ROIS = [
        # top board
        (cx - w//2, cy - h//2 - dy,
         cx + w//2, cy + h//2 - dy),

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

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        for board_id, roi in enumerate(ROIS):

            x1, y1, x2, y2 = roi

            # Mask out outside ROI
            mask = np.zeros_like(gray, dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

            # apply mask (keep full image size!)
            masked_img = cv.bitwise_and(gray, gray, mask=mask)

            ret, corners = cv.findChessboardCorners(masked_img, CHESSBOARD, None)

            if not ret:
                continue

            corners2 = cv.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)

            objpoints[board_id].append(objp_base.copy())
            imgpoints[board_id].append(corners2)

    img_size = gray.shape[::-1]

    # Flatten
    obj_flat = []
    img_flat = []

    for b in range(len(objpoints)):
        for i in range(len(objpoints[b])):
            obj_flat.append(objpoints[b][i])
            img_flat.append(imgpoints[b][i])

    return obj_flat, img_flat, img_size

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

# TODO change for Pi
# TODO read in from file
def prep_camera_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_device', '-v', type=str, default='0')
    args = parser.parse_args()

    return args



# TODO could make timing better
def save_calib_video(args, calib_time = 1.0, calib_folder = "outputs/baseline"):
    # Take a 2 second video and save frames - use VideoCapture/imwrite for high quality
    try:
        camera_device = cv.VideoCapture(int(args.video_device))
    except ValueError:
        camera_device = cv.VideoCapture(args.video_device)

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



def save_exp_video(args, exp_time = 5.0, baseline_folder = "outputs/baseline"):
    # Take a 30 second video and save frames - use VideoCapture/imwrite for high quality
    try:
        camera_device = cv.VideoCapture(int(args.video_device))
    except ValueError:
        camera_device = cv.VideoCapture(args.video_device)

    if not camera_device.isOpened():
        print('Could not access camera')
        sys.exit()

    frames = []
    start_time = time.time()

    # Run for experiment time and append every frame
    while time.time() - start_time < exp_time:
        ret, frame = camera_device.read()
        if not ret:
            continue

        frames.append(frame)

    camera_device.release()

    # Save video to calibration folder TODO check this
    shutil.rmtree(baseline_folder, ignore_errors=True)
    os.makedirs(baseline_folder, exist_ok=True)

    for i, frame in enumerate(frames):
        cv.imwrite(f"{baseline_folder}/frame_{i:04d}.jpeg", frame)



# Cycle through frames, do pose estimation
# Save all pose estimates to file
# Delete frames
# (test first on calibration images)


def process_baseline_data(objpoints_3boards, mtx, dist, ROIS, MAX_BOARDS = 3, CHESSBOARD = (5, 3), baseline_folder = "outputs/calibration", pose_folder = "outputs/baseline_pose"):
    # TODO Read in images 
    images = glob.glob(f"{baseline_folder}/*.jpeg")

    # TODO do over all the ROIs

    # TODO Test on an image and save each with pose estimation detection
    
    axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]]).reshape(-1,3)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        for board_id, roi in enumerate(ROIS):

            x1, y1, x2, y2 = roi

            # Mask out outside ROI
            mask = np.zeros_like(gray, dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

            # apply mask (keep full image size!)
            working_img = cv.bitwise_and(gray, gray, mask=mask)

            ret, corners = cv.findChessboardCorners(
                working_img, CHESSBOARD,
                cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
            )

            # Skip if no image
            # TODO blank out pose estimation
            if not ret:
                break

            # Refine
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

            # Pose estimation
            obj_model = objpoints_3boards[board_id]

            success, rvec, tvec = cv.solvePnP(obj_model, corners2, mtx, dist)

            # TODO Comment out below
            # Drawing board onto image (debugging)
            if success:  
                print(f"\nImage: {fname}, Board #{board_id+1}")
                print("rvec:\n", rvec)
                print("tvec:\n", tvec)

                imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)

                # Draw corners with thicker visualisation
                cv.drawChessboardCorners(img, CHESSBOARD, corners2, False)

                corners_int = corners2.astype(int)

                for i in range(len(corners_int) - 1):
                    cv.line(img, tuple(corners_int[i][0]), tuple(corners_int[i+1][0]), (0, 255, 255), 13)

                # Label board ID
                label = board_id + 1
                corner = tuple(corners2[0].ravel().astype(int))

                cv.putText(img, f"{label}", (corner[0] + 10, corner[1] - 30), cv.FONT_HERSHEY_SIMPLEX,
                    5, (0, 0, 255), 6, cv.LINE_AA)

        # cv.imshow('Multi Pose', img)
        
        base = os.path.basename(fname)
        name, _ = os.path.splitext(base)

        output_name = os.path.join(pose_folder, f"{name}_multi_pose.png")

        success = cv.imwrite(output_name, img)

        print(f"SAVE {output_name} -> {success}")

        output_name = f"multi_pose_{fname.split('/')[-1]}"
        cv.imwrite(output_name, img)


# Working
# def process_baseline_data(objpoints_3boards, mtx, dist, MAX_BOARDS = 3, CHESSBOARD = (5, 3), baseline_folder = "outputs/calibration", pose_folder = "outputs/baseline_pose"):
#     # TODO Read in images 
#     images = glob.glob(f"{baseline_folder}/*.jpeg")

#     # TODO do over all the ROIs

#     # TODO Test on an image and save each with pose estimation detection
    
#     axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]]).reshape(-1,3)
#     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#     for fname in images:
#         img = cv.imread(fname)
#         gray_full = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#         working_img = gray_full.copy()
#         boards_found = 0

#         while boards_found < MAX_BOARDS:
#             ret, corners = cv.findChessboardCorners(
#                 working_img, CHESSBOARD,
#                 cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
#             )

#             # Skip if no image
#             if not ret:
#                 break

#             # Refine
#             corners2 = cv.cornerSubPix(gray_full, corners, (11,11), (-1,-1), criteria)

#             # Pose estimation
#             obj_model = objpoints_3boards[boards_found]

#             success, rvec, tvec = cv.solvePnP(obj_model, corners2, mtx, dist)

#             # TODO Comment out below
#             # Drawing board onto image (debugging)
#             if success:  
#                 print(f"\nImage: {fname}, Board #{boards_found+1}")
#                 print("rvec:\n", rvec)
#                 print("tvec:\n", tvec)

#                 imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)

#                 # Draw corners with thicker visualisation
#                 cv.drawChessboardCorners(img, CHESSBOARD, corners2, False)

#                 corners_int = corners2.astype(int)

#                 for i in range(len(corners_int) - 1):
#                     cv.line(img, tuple(corners_int[i][0]), tuple(corners_int[i+1][0]), (0, 255, 255), 13)

#                 # Label board ID
#                 label = boards_found + 1
#                 corner = tuple(corners2[0].ravel().astype(int))

#                 cv.putText(img, f"{label}", (corner[0] + 10, corner[1] - 30), cv.FONT_HERSHEY_SIMPLEX,
#                     5, (0, 0, 255), 6, cv.LINE_AA)

#             # Mask out detected board 
#             # TODO replace with ROI bounds
#             hull = cv.convexHull(corners.astype(np.int32))
#             cv.fillConvexPoly(working_img, hull, 0)

#             boards_found += 1
#         # cv.imshow('Multi Pose', img)
        
#         base = os.path.basename(fname)
#         name, _ = os.path.splitext(base)

#         output_name = os.path.join(pose_folder, f"{name}_multi_pose.png")

#         success = cv.imwrite(output_name, img)

#         print(f"SAVE {output_name} -> {success}")

#         output_name = f"multi_pose_{fname.split('/')[-1]}"
#         cv.imwrite(output_name, img)



