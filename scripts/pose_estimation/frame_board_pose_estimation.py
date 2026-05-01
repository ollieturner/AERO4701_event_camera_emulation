import numpy as np
import cv2 as cv
import glob

# ==============================
# PARAMETERS
# ==============================
# CHESSBOARD = (10, 7)
CHESSBOARD = (5, 3)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
MAX_BOARDS = 3
# SQUARE_SIZE = 0.024  # metres (example: 24 mm per square)
SQUARE_SIZE = 0.00225  # metres (example: 24 mm per square)


# Prepare object points
objp = np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

print("Starting program...")

images = glob.glob('scripts/pose_estimation/board_imgs/*.png')

# ==============================
# HELPER: DRAW AXES
# ==============================
def draw_axes(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 3)
    img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3)
    img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 3)
    return img

axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]]).reshape(-1,3)

# ==============================
# STEP 1: CALIBRATION (single-board assumption)
# ==============================
objpoints = []
imgpoints = []

print("Calibration pass...")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

ret, mtx, dist, _, _ = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Camera matrix:\n", mtx)

# ==============================
# STEP 2: MULTI-BOARD DETECTION
# ==============================
print("Detecting multiple checkerboards...")

for fname in images:
    img = cv.imread(fname)
    gray_full = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    working_img = gray_full.copy()
    boards_found = 0

    while boards_found < MAX_BOARDS:
        ret, corners = cv.findChessboardCorners(
            working_img, CHESSBOARD,
            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
        )

        if not ret:
            break

        # Refine
        corners2 = cv.cornerSubPix(
            gray_full, corners, (11,11), (-1,-1), criteria
        )

        # Pose estimation
        success, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)

        if success:
            print(f"\nImage: {fname}, Board #{boards_found+1}")
            print("rvec:\n", rvec)
            print("tvec:\n", tvec)

            imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)
            img = draw_axes(img, corners2, imgpts)

            cv.drawChessboardCorners(img, CHESSBOARD, corners2, True)

        # ==============================
        # MASK OUT DETECTED BOARD
        # ==============================
        hull = cv.convexHull(corners.astype(np.int32))
        cv.fillConvexPoly(working_img, hull, 0)

        boards_found += 1
    cv.imshow('Multi Pose', img)

# Keep window open until user closes or presses ESC
while True:
    key = cv.waitKey(20) & 0xFF

    # ESC key to exit
    if key == 27:
        break

    # If window is closed manually
    if cv.getWindowProperty('Multi Pose', cv.WND_PROP_VISIBLE) < 1:
        break

cv.destroyAllWindows()





#     cv.imshow('Multi Pose', img)
#     cv.waitKey(500)



# cv.destroyAllWindows()





## MULTI 
# import numpy as np
# import cv2 as cv
# import glob

# # ==============================
# # PARAMETERS
# # ==============================
# CHESSBOARD = (10, 7)   # inner corners
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # Prepare object points (Z=0 plane)
# objp = np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

# # Storage
# objpoints = []
# imgpoints = []

# images = glob.glob('checkerboard_imgs/*.jpg')

# # ==============================
# # STEP 1: DETECT CORNERS
# # ==============================
# print("Detecting checkerboards...")

# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     ret, corners = cv.findChessboardCorners(
#         gray, CHESSBOARD,
#         cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
#     )

#     if ret:
#         objpoints.append(objp)

#         corners2 = cv.cornerSubPix(
#             gray, corners, (11,11), (-1,-1), criteria
#         )
#         imgpoints.append(corners2)

#         cv.drawChessboardCorners(img, CHESSBOARD, corners2, ret)
#         cv.imshow('Corners', img)
#         cv.waitKey(200)

# cv.destroyAllWindows()

# # ==============================
# # STEP 2: CALIBRATION
# # ==============================
# print("Calibrating camera...")

# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
#     objpoints, imgpoints, gray.shape[::-1], None, None
# )

# print("Camera matrix:\n", mtx)
# print("Distortion:\n", dist)

# # ==============================
# # FUNCTION: DRAW AXES
# # ==============================
# def draw_axes(img, corners, imgpts):
#     corner = tuple(corners[0].ravel().astype(int))

#     img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 3) # X (blue)
#     img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3) # Y (green)
#     img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 3) # Z (red)

#     return img

# # Axis length (in checkerboard units)
# axis = np.float32([
#     [3,0,0],
#     [0,3,0],
#     [0,0,-3]
# ]).reshape(-1,3)

# # ==============================
# # STEP 3: POSE ESTIMATION
# # ==============================
# print("Estimating pose...")

# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, None)

#     if ret:
#         corners2 = cv.cornerSubPix(
#             gray, corners, (11,11), (-1,-1), criteria
#         )

#         # --- Solve PnP ---
#         success, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)

#         if success:
#             print("\nImage:", fname)
#             print("rvec:\n", rvec)
#             print("tvec:\n", tvec)

#             # Project 3D axes to image
#             imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)

#             # Draw axes
#             img = draw_axes(img, corners2, imgpts)

#             cv.imshow('Pose', img)
#             cv.waitKey(500)

# cv.destroyAllWindows()