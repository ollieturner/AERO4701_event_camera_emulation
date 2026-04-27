# Assumes horizontal, vertical - not rotational invariant

import numpy as np
import cv2 as cv
import yaml

# -----------------------------
# Settings
# -----------------------------
CHESSBOARD = (10, 7)

objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []


def classify_lines_by_angle(lines):
    """
    Classify Hough lines into two dominant orientation groups (rotation-invariant).

    Args:
        lines: output of cv.HoughLinesP

    Returns:
        group1, group2: two lists of lines (each line = (x1,y1,x2,y2))
    """

    if lines is None or len(lines) < 2:
        return [], []

    angles = []
    lines_list = []

    # -----------------------------
    # Compute angles
    # -----------------------------
    for l in lines:
        x1, y1, x2, y2 = l[0]

        angle = np.arctan2(y2 - y1, x2 - x1)
        angle = angle % np.pi  # normalize to [0, π)

        angles.append(angle)
        lines_list.append((x1, y1, x2, y2))

    angles = np.array(angles)

    # -----------------------------
    # Split into 2 groups (simple clustering)
    # -----------------------------
    median_angle = np.median(angles)

    group1 = []
    group2 = []

    for angle, line in zip(angles, lines_list):
        if abs(angle - median_angle) < np.pi / 4:
            group1.append(line)
        else:
            group2.append(line)

    return group1, group2


# -----------------------------
# Helper: extract grid intersections from edges
# -----------------------------
def extract_grid_points(edges, grid_size):
    rows, cols = grid_size

    # # detect lines
    # lines = cv.HoughLinesP(
    #     edges,
    #     1,
    #     np.pi / 180,
    #     threshold=80,
    #     minLineLength=30,
    #     maxLineGap=10
    # )
    # detect lines
    lines = cv.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80, # lower is more lines
        minLineLength=30, # lower could be broken lines
        maxLineGap=10 # allows connecting broken edges
    )

    if lines is None:
        return None, None, None

    horizontals = []
    verticals = []

    # classify lines
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) > abs(dy):
            horizontals.append((x1, y1, x2, y2))
        else:
            verticals.append((x1, y1, x2, y2))
       

    if len(horizontals) < rows or len(verticals) < cols:
        return None, None, None

    # take strongest lines only
    horizontals = horizontals[:rows]
    verticals = verticals[:cols]

    points = []

    # compute intersections
    for h in horizontals:
        for v in verticals:

            x1, y1, x2, y2 = h
            x3, y3, x4, y4 = v

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

            if abs(denom) < 1e-6:
                continue

            px = ((x1*y2 - y1*x2)*(x3 - x4) -
                  (x1 - x2)*(x3*y4 - y3*x4)) / denom

            py = ((x1*y2 - y1*x2)*(y3 - y4) -
                  (y1 - y2)*(x3*y4 - y3*x4)) / denom

            points.append([px, py])

    if len(points) != rows * cols:
        return None, None, None

    points = np.array(points, dtype=np.float32)

    # sort into grid (top-left origin assumption)
    points = points[np.lexsort((points[:, 0], points[:, 1]))]

    # return points.reshape(-1, 1, 2)
    return points.reshape(-1, 1, 2), horizontals, verticals



# -----------------------------
# Video
# -----------------------------
cap = cv.VideoCapture('event.mp4')

if not cap.isOpened():
    raise Exception("Could not open video")

fps = cap.get(cv.CAP_PROP_FPS)
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
# out = cv.VideoWriter('event_with_checkerboard.mp4', fourcc, fps, (w, h))

out_lines = cv.VideoWriter('event_with_lines.mp4', fourcc, fps, (w, h))
out_points = cv.VideoWriter('event_with_points.mp4', fourcc, fps, (w, h))

# -----------------------------
# Process frames
# -----------------------------
print("Processing...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Finished processing")
        break

    raw = frame.copy()

    # -----------------------------
    # Edge extraction (event-friendly)
    # -----------------------------
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # edges = cv.Canny(gray, 40, 120)
    edges = cv.Canny(gray, 20, 80)


    # -----------------------------
    # GRID INTERSECTION DETECTION
    # -----------------------------
    # corners = extract_grid_points(edges, CHESSBOARD)
    corners, horizontals, verticals = extract_grid_points(edges, CHESSBOARD)

    line_frame = raw.copy()

    if horizontals is not None:
        for (x1, y1, x2, y2) in horizontals:
            cv.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # blue

    if verticals is not None:
        for (x1, y1, x2, y2) in verticals:
            cv.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # red

    out_lines.write(line_frame)

    points_frame = raw.copy()

    if corners is not None:
        print("Detected edge-based grid!")

        objpoints.append(objp)
        imgpoints.append(corners)

        for p in corners:
            x, y = int(p[0][0]), int(p[0][1])
            cv.circle(points_frame, (x, y), 3, (0, 255, 0), -1)

    out_points.write(points_frame)
            
    # if corners is not None:

    #     print("Detected edge-based grid!")

    #     objpoints.append(objp)
    #     imgpoints.append(corners)

        # for p in corners:
        #     x, y = int(p[0][0]), int(p[0][1])
        #     cv.circle(raw, (x, y), 3, (0, 255, 0), -1)

    # out.write(raw)

# # -----------------------------
# # Calibration
# # -----------------------------
# if len(objpoints) > 5:
#     print("Calibrating...")

#     ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
#         objpoints,
#         imgpoints,
#         (w, h),
#         None,
#         None
#     )

#     print("K:\n", mtx)

#     data = {"K": mtx.tolist()}

#     with open('event_data.yaml', 'w') as f:
#         yaml.dump(data, f, default_flow_style=False, sort_keys=False)