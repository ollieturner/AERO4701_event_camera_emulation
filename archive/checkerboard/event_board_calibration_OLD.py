# FAST corners?



import numpy as np
import cv2 as cv
import yaml

# -----------------------------
# Settings
# -----------------------------
CHESSBOARD = (10, 7)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# -----------------------------
# Open video
# -----------------------------
cap = cv.VideoCapture('event.mp4')

if not cap.isOpened():
    raise Exception("Could not open video")

fps = cap.get(cv.CAP_PROP_FPS)
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('event_with_checkerboard.mp4', fourcc, fps, (w, h))

# -----------------------------
# Process frames
# -----------------------------
print("Processing...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Finished processing")

        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    # raw_frame = frame.copy()
    # frame = raw_frame.astype(np.float32)

    # # red - blue polarity map
    # event_img = frame[:, :, 2] - frame[:, :, 0]
    # event_img = cv.normalize(event_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # gray = event_img

    # edges = cv.Canny(gray, 50, 150)

    # contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    ret_corners, corners = cv.findChessboardCorners(
        gray,
        CHESSBOARD,
        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret_corners:
        print("Found a checkerboard!")
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(frame, CHESSBOARD, corners2, ret_corners)

    out.write(frame)


# Calibration 
if len(objpoints) > 0:
    print("Calibrating...")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    data = {"K": mtx.tolist()}

    with open('event_data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)

