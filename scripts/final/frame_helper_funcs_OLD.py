import numpy as np
import cv2 as cv

# TODO: Check this
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


def detect_chessboards_calib(images, objpoints, imgpoints, criteria, CHESSBOARD = (5, 3), SQUARE_SIZE = 0.00225):
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



# Don't use in final?
def draw_axes(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 3)
    img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3)
    img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 3)
    return img

axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]]).reshape(-1,3)