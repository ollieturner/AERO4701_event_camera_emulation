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
cap = cv.VideoCapture('original.mp4')

if not cap.isOpened():
    raise Exception("Could not open video")

fps = cap.get(cv.CAP_PROP_FPS)
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('original_with_chessboard.mp4', fourcc, fps, (w, h))

# -----------------------------
# Process frames
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    ret_corners, corners = cv.findChessboardCorners(
        gray,
        CHESSBOARD,
        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret_corners:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(frame, CHESSBOARD, corners2, ret_corners)

    out.write(frame)

#     cv.imshow('frame', frame)
#     if cv.waitKey(1) & 0xFF == 27:  # ESC to quit
#         break

# # -----------------------------
# # Cleanup
# # -----------------------------
# cap.release()
# out.release()
# cv.destroyAllWindows()

# Calibration 
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    data = {"K": mtx.tolist()}

    with open('data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)





# import numpy as np
# import cv2 as cv
# import glob
# import yaml

# # Checkerboard
# CHESSBOARD = (10,7)

# # width
# SCREENWIDTH = 1000

# # termination criteria
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((CHESSBOARD[0]*CHESSBOARD[1],3), np.float32)
# objp[:,:2] = np.mgrid[0:CHESSBOARD[0],0:CHESSBOARD[1]].T.reshape(-1,2)


# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.

# # images = glob.glob('checkerboard_imgs/*.jpg')

# # cv.namedWindow('img', cv.WINDOW_NORMAL)
# # cv.resizeWindow('img', 600, 400)




# # -----------------------------
# # Open video
# # -----------------------------
# cap = cv.VideoCapture('original.mp4')

# if not cap.isOpened():
#     raise Exception("Could not open video")

# # Get video properties
# fps = cap.get(cv.CAP_PROP_FPS)
# w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# # -----------------------------
# # Setup VideoWriter
# # -----------------------------
# fourcc = cv.VideoWriter_fourcc(*'mp4v')
# out = cv.VideoWriter('original_with_tags.mp4', fourcc, fps, (w, h))


# try:
#     while True:
#         ret, fname = cap.read()
#         if not ret:
#             break


# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     # Find the chess board corners
#     # ret, corners = cv.findChessboardCorners(gray, (10, 7), None)
#     ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)

#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)

#         corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners2)

#         # Draw and display the corners
#         cv.drawChessboardCorners(img, CHESSBOARD, corners2, ret)

#         cv.imshow('img', img)
#         cv.waitKey(100)

# cv.destroyAllWindows()


# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)



# data = {
#     "K": mtx.tolist()
# }

# with open('data.yaml', 'w') as outfile:
#     yaml.dump(data, outfile, default_flow_style=True, sort_keys=False)
