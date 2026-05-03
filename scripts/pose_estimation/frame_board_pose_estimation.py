import numpy as np
import cv2 as cv
import glob

import os

# create output folder once (put near top of script)
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# PARAMETERS
# ==============================
# CHESSBOARD = (10, 7)
CHESSBOARD = (5, 3)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
MAX_BOARDS = 3
# SQUARE_SIZE = 0.024  # metres (example: 24 mm per square)
SQUARE_SIZE = 0.00225  # metres (example: 24 mm per square)


L = 0.034511
h = np.sqrt(3)/2 * L

board_centres = np.array([
    [0,  2*h/3, 0],        # top
    [-L/2, -h/3, 0],       # bottom-left
    [ L/2, -h/3, 0]        # bottom-right
])

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


objpoints_3boards = []

CHESSBOARD = (5, 3)

grid = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
grid = np.hstack([grid * SQUARE_SIZE, np.zeros((grid.shape[0], 1))])

for centre in board_centres:
    R = rotation_towards(centre)

    rotated = (R @ grid.T).T
    translated = rotated + centre

    objpoints_3boards.append(translated.astype(np.float32))


# # Prepare object points
# objp = np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

print("Starting program...")

images = glob.glob('scripts/pose_estimation/board_imgs/*.jpeg')

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
        # objpoints.append(objp)
        objp_calib = np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
        objp_calib[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE
        objpoints.append(objp_calib)

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
        # success, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)
        obj_model = objpoints_3boards[boards_found]

        success, rvec, tvec = cv.solvePnP(
            obj_model,
            corners2,
            mtx,
            dist
        )

        if success:
            print(f"\nImage: {fname}, Board #{boards_found+1}")
            print("rvec:\n", rvec)
            print("tvec:\n", tvec)

            imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)
            # img = draw_axes(img, corners2, imgpts)

            # cv.drawChessboardCorners(img, CHESSBOARD, corners2, True)
            # draw corners (thicker visualisation)
            cv.drawChessboardCorners(img, CHESSBOARD, corners2, False)

            corners_int = corners2.astype(int)

            for i in range(len(corners_int) - 1):
                cv.line(img,
                        tuple(corners_int[i][0]),
                        tuple(corners_int[i+1][0]),
                        (0, 255, 255),
                        13)

            # label board ID
            label = boards_found + 1
            corner = tuple(corners2[0].ravel().astype(int))

            cv.putText(
                img,
                f"{label}",
                (corner[0] + 10, corner[1] - 30),
                cv.FONT_HERSHEY_SIMPLEX,
                5,
                (0, 0, 255),
                6,
                cv.LINE_AA
            )


        # ==============================
        # MASK OUT DETECTED BOARD
        # ==============================
        hull = cv.convexHull(corners.astype(np.int32))
        cv.fillConvexPoly(working_img, hull, 0)

        boards_found += 1
    # cv.imshow('Multi Pose', img)
    
    base = os.path.basename(fname)
    name, _ = os.path.splitext(base)

    output_name = os.path.join(output_dir, f"{name}_multi_pose.png")

    success = cv.imwrite(output_name, img)

    print(f"[SAVE] {output_name} -> {success}")

    # output_name = f"multi_pose_{fname.split('/')[-1]}"
    # cv.imwrite(output_name, img)


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