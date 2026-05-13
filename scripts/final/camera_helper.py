import numpy as np
import cv2 as cv
import time
import shutil
import os


def setup_directories():
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    calib_folder = "outputs/calibration"
    os.makedirs(calib_folder, exist_ok=True)

    return output_dir, calib_folder


def setup_calib_parameters():
    CHESSBOARD = (5, 3)
    MAX_BOARDS = 3
    SQUARE_SIZE = 0.00225       # in metres
    L = 0.034511                # for triangular positions of boards
    MAX_CALIB_ATTEMPTS = 5
    
    return CHESSBOARD, MAX_BOARDS, SQUARE_SIZE, L, MAX_CALIB_ATTEMPTS

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


def detect_cboard_calib(images, CHESSBOARD = (5, 3), SQUARE_SIZE = 0.00225):
    print("Calibrating camera...")

    objpoints = []
    imgpoints = []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # TODO replace with findChessboardCornersSB
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, None)

        if ret:
            objp_calib = np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
            objp_calib[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE
            objpoints.append(objp_calib)
            # TODO Reduce from 11, 11?
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
        
    img_size = gray.shape[::-1]

    return objpoints, imgpoints, img_size

# TODO coule make timining better
def save_calib_video(calib_time = 2.0, calib_folder = "outputs/calibration"):
    # Take a 2 second video and save frames
    cap = cv.VideoCapture(0)

    frames = []
    start_time = time.time()

    while time.time() - start_time < calib_time:
        ret, frame = cap.read()
        if not ret:
            continue

        frames.append(frame)

    cap.release()

    # Save video to calibration folder
    shutil.rmtree(calib_folder, ignore_errors=True)
    os.makedirs(calib_folder, exist_ok=True)

    for i, frame in enumerate(frames):
        cv.imwrite(f"{calib_folder}/frame_{i:04d}.jpeg", frame)
