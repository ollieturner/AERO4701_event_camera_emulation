import numpy as np
from sklearn.neighbors import NearestNeighbors


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

# FAST detector
fast = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

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
out = cv.VideoWriter('event_with_chessboard.mp4', fourcc, fps, (w, h))


# -----------------------------
# Helper: convert FAST keypoints → grid-like corners
# -----------------------------
# def knn_cluster_points(keypoints, k=3, max_dist=1): # 25
#     """
#     Cluster FAST keypoints using KNN.
#     Keeps clusters of size 3 where points are close enough.

#     Args:
#         keypoints: FAST keypoints
#         k: number of neighbors (fixed at 3 as requested)
#         max_dist: distance threshold (pixels)

#     Returns:
#         filtered clustered points (N,1,2)
#     """

#     if len(keypoints) < k:
#         return None

#     pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)

#     # -----------------------------
#     # Fit KNN
#     # -----------------------------
#     nn = NearestNeighbors(n_neighbors=k)
#     nn.fit(pts)

#     distances, indices = nn.kneighbors(pts)

#     used = set()
#     clusters = []

#     # -----------------------------
#     # Build clusters of 3 points
#     # -----------------------------
#     for i in range(len(pts)):

#         if i in used:
#             continue

#         neigh_idx = indices[i]

#         cluster = pts[neigh_idx]

#         # -----------------------------
#         # Check cluster compactness
#         # -----------------------------
#         dmat = np.linalg.norm(cluster - cluster.mean(axis=0), axis=1)

#         if np.max(dmat) > max_dist:
#             continue  # reject loose cluster

#         # mark used
#         for idx in neigh_idx:
#             used.add(idx)

#         clusters.append(cluster)

#     if len(clusters) == 0:
#         return None

#     # -----------------------------
#     # Flatten clusters
#     # -----------------------------
#     clusters = np.vstack(clusters).astype(np.float32)

#     return clusters.reshape(-1, 1, 2)

def knn_cluster_points(keypoints, k=3, max_dist=5):
    """
    KNN clustering of FAST keypoints.
    Each valid 3-point cluster is replaced by its centroid.
    """

    if len(keypoints) < k:
        return None

    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(pts)

    distances, indices = nn.kneighbors(pts)

    used = set()
    centroids = []

    for i in range(len(pts)):

        if i in used:
            continue

        neigh_idx = indices[i]
        cluster = pts[neigh_idx]

        # compute spread
        center = cluster.mean(axis=0)
        dists = np.linalg.norm(cluster - center, axis=1)

        # reject loose clusters
        if np.max(dists) > max_dist:
            continue

        # mark points as used
        for idx in neigh_idx:
            used.add(idx)

        # -----------------------------
        # KEEP ONLY MEAN (THIS IS THE KEY CHANGE)
        # -----------------------------
        centroids.append(center)

    if len(centroids) == 0:
        return None

    centroids = np.array(centroids, dtype=np.float32)

    return centroids.reshape(-1, 1, 2)


# -----------------------------
# Process frames
# -----------------------------
print("Processing...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Finished processing")
        break

    raw_frame = frame.copy()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # -----------------------------
    # Edge stabilisation (IMPORTANT for event-like input)
    # -----------------------------
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    edges = cv.Canny(gray, 40, 120)

    # -----------------------------
    # FAST corners on edges
    # -----------------------------
    keypoints = fast.detect(edges, None)

    # Draw FAST points (debug)
    # debug = cv.drawKeypoints(raw_frame, keypoints, None, color=(0, 255, 0))
    debug = raw_frame

    # -----------------------------
    # Convert FAST → pseudo grid
    # -----------------------------
    corners = knn_cluster_points(keypoints) #  k=3, max_dist=25)

    if corners is not None and len(corners) >= CHESSBOARD[0] * CHESSBOARD[1]:

        print("Detected grid-like structure")

        corners = corners[:CHESSBOARD[0] * CHESSBOARD[1]]

        objpoints.append(objp)
        imgpoints.append(corners)

        cv.drawChessboardCorners(debug, CHESSBOARD, corners, True)

    out.write(debug)

# # -----------------------------
# # Calibration
# # -----------------------------
# if len(objpoints) > 5:
#     print("Calibrating...")

#     ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
#         objpoints,
#         imgpoints,
#         (w, h),
#         None,
#         None
#     )

#     print("K:\n", K)

#     data = {"K": K.tolist()}

#     with open('event_data.yaml', 'w') as f:
#         yaml.dump(data, f, default_flow_style=False, sort_keys=False)











# import numpy as np
# import cv2 as cv
# import yaml

# # -----------------------------
# # Settings
# # -----------------------------
# CHESSBOARD = (10, 7)

# objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

# objpoints = []
# imgpoints = []

# # FAST detector
# fast = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

# # -----------------------------
# # Video
# # -----------------------------
# cap = cv.VideoCapture('event.mp4')

# if not cap.isOpened():
#     raise Exception("Could not open video")

# fps = cap.get(cv.CAP_PROP_FPS)
# w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv.VideoWriter_fourcc(*'mp4v')
# out = cv.VideoWriter('new_event_with_chessboard.mp4', fourcc, fps, (w, h))


# # -----------------------------
# # Helper: convert FAST keypoints → grid-like corners
# # -----------------------------
# def filter_grid_points(keypoints, grid_size):
#     rows, cols = grid_size  # (10,7)

#     if len(keypoints) < rows * cols:
#         return None

#     pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)

#     # -----------------------------
#     # Step 1: sort by Y (row clustering)
#     # -----------------------------
#     pts = pts[np.argsort(pts[:, 1])]

#     row_groups = np.array_split(pts, rows)

#     filtered = []

#     # -----------------------------
#     # Step 2: within each row, sort by X (column structure)
#     # -----------------------------
#     for row in row_groups:
#         if len(row) < cols:
#             continue

#         row = row[np.argsort(row[:, 0])]
#         row = row[:cols]  # keep only strongest structure

#         filtered.append(row)

#     if len(filtered) != rows:
#         return None

#     # -----------------------------
#     # Step 3: flatten back to grid
#     # -----------------------------
#     filtered = np.array(filtered, dtype=np.float32)
#     filtered = filtered.reshape(-1, 2)

#     return filtered.reshape(-1, 1, 2)

# # -----------------------------
# # Process frames
# # -----------------------------
# print("Processing...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Finished processing")
#         break

#     raw_frame = frame.copy()

#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # -----------------------------
#     # Edge stabilisation (IMPORTANT for event-like input)
#     # -----------------------------
#     gray = cv.GaussianBlur(gray, (5, 5), 0)
#     gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

#     edges = cv.Canny(gray, 40, 120)

#     # -----------------------------
#     # FAST corners on edges
#     # -----------------------------
#     keypoints = fast.detect(edges, None)

#     # Draw FAST points (debug)
#     debug = cv.drawKeypoints(raw_frame, keypoints, None, color=(0, 255, 0))

#     # -----------------------------
#     # Convert FAST → pseudo grid
#     # -----------------------------
#     corners = filter_grid_points(keypoints, CHESSBOARD)

#     if corners is not None and len(corners) >= CHESSBOARD[0] * CHESSBOARD[1]:

#         print("Detected grid-like structure")

#         corners = corners[:CHESSBOARD[0] * CHESSBOARD[1]]

#         objpoints.append(objp)
#         imgpoints.append(corners)

#         cv.drawChessboardCorners(debug, CHESSBOARD, corners, True)

#     out.write(debug)

# # -----------------------------
# # Calibration
# # -----------------------------
# if len(objpoints) > 5:
#     print("Calibrating...")

#     ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
#         objpoints,
#         imgpoints,
#         (w, h),
#         None,
#         None
#     )

#     print("K:\n", K)

#     data = {"K": K.tolist()}

#     with open('event_data.yaml', 'w') as f:
#         yaml.dump(data, f, default_flow_style=False, sort_keys=False)






# import numpy as np
# import cv2 as cv
# import yaml

# # -----------------------------
# # Settings
# # -----------------------------
# CHESSBOARD = (10, 7)

# objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

# objpoints = []
# imgpoints = []

# # FAST detector
# fast = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

# # -----------------------------
# # Video
# # -----------------------------
# cap = cv.VideoCapture('event.mp4')

# if not cap.isOpened():
#     raise Exception("Could not open video")

# fps = cap.get(cv.CAP_PROP_FPS)
# w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv.VideoWriter_fourcc(*'mp4v')
# out = cv.VideoWriter('event_with_chessboard.mp4', fourcc, fps, (w, h))


# # -----------------------------
# # Helper: convert FAST keypoints → grid-like corners
# # -----------------------------
# def fast_to_grid(keypoints, pattern_size):
#     if len(keypoints) < pattern_size[0] * pattern_size[1]:
#         return None

#     pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)

#     # sort by y then x (rough grid ordering)
#     pts = pts[np.lexsort((pts[:, 0], pts[:, 1]))]

#     return pts.reshape(-1, 1, 2)


# # -----------------------------
# # Process frames
# # -----------------------------
# print("Processing...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Finished processing")
#         break

#     raw_frame = frame.copy()

#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # -----------------------------
#     # Edge stabilisation (IMPORTANT for event-like input)
#     # -----------------------------
#     gray = cv.GaussianBlur(gray, (5, 5), 0)
#     gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

#     edges = cv.Canny(gray, 40, 120)

#     # -----------------------------
#     # FAST corners on edges
#     # -----------------------------
#     keypoints = fast.detect(edges, None)

#     # Draw FAST points (debug)
#     debug = cv.drawKeypoints(raw_frame, keypoints, None, color=(0, 255, 0))

#     # -----------------------------
#     # Convert FAST → pseudo grid
#     # -----------------------------
#     corners = fast_to_grid(keypoints, CHESSBOARD)

#     if corners is not None and len(corners) >= CHESSBOARD[0] * CHESSBOARD[1]:

#         print("Detected grid-like structure")

#         corners = corners[:CHESSBOARD[0] * CHESSBOARD[1]]

#         objpoints.append(objp)
#         imgpoints.append(corners)

#         cv.drawChessboardCorners(debug, CHESSBOARD, corners, True)

#     out.write(debug)

# # -----------------------------
# # Calibration
# # -----------------------------
# if len(objpoints) > 5:
#     print("Calibrating...")

#     ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
#         objpoints,
#         imgpoints,
#         (w, h),
#         None,
#         None
#     )

#     print("K:\n", K)

#     data = {"K": K.tolist()}

#     with open('event_data.yaml', 'w') as f:
#         yaml.dump(data, f, default_flow_style=False, sort_keys=False)