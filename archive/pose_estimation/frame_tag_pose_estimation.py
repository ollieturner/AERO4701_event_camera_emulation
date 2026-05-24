import cv2
import numpy as np
import glob
from dt_apriltags import Detector
import yaml

# Initialise detector
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


# Define known tag geometry (world frame)
# - Choose origin, then tag corners relative to that (in m)
d = 0.2  # spacing between tag origins
tag_size = 0.065 # load it in from yaml 


tag_layout = {
    0: np.array([[0, d, 0],
                 [tag_size, d, 0],
                 [tag_size, d + tag_size, 0],
                 [0, d + tag_size, 0]], dtype=np.float32),

    1: np.array([[d, d, 0],
                 [d + tag_size, d, 0],
                 [d + tag_size, d + tag_size, 0],
                 [d, d + tag_size, 0]], dtype=np.float32),

    2: np.array([[0, 0, 0],
                 [tag_size, 0, 0],
                 [tag_size, tag_size, 0],
                 [0, tag_size, 0]], dtype=np.float32)
}

# Initialise calibration storage
obj_points = []
img_points = []

# Cycle through calibration images and store corner points
# Optional: visualisation
image_files = glob.glob("calib_images/*.png")

for fname in image_files:
    # Open image in grayscale
    gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue

    tags = at_detector.detect(gray)

    # Extract information for one tag then visualise
    for tag in tags:
        if tag.tag_id in tag_layout:
            # Extract 3D and 2D corners 
            obj_points.append(tag_layout[tag.tag_id])          # 3D corners
            img_points.append(tag.corners.astype(np.float32))  # 2D corners

            # Visualisation
            colour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            corners = tag.corners.astype(int)

            for i in range(4):
                cv2.line(colour_img,
                         tuple(corners[i - 1]),
                         tuple(corners[i]),
                         (0, 255, 0), 2)

            cv2.putText(colour_img,
                        str(tag.tag_id),
                        (corners[0][0] + 10, corners[0][1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)

            cv2.imshow("Detection", colour_img)
            cv2.waitKey(100)

cv2.destroyAllWindows()

# Calibrate camera with calibraion pointed
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points,
    img_points,
    gray.shape[::-1],
    None,
    None
)

# Output results
print("\n===== INTRINSIC MATRIX K =====")
print(K)


# # Write out K to yaml 
# print(type(K))

# data = {
#     "K": K.tolist(), 
#     "tag_size": tag_size
# }

# with open('data.yaml', 'w') as outfile:
#     yaml.dump(data, outfile, default_flow_style=True, sort_keys=False)



# Extract intrinsics
fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

camera_params = (fx, fy, cx, cy)

# # Tag size (in metres)
# tag_size = cam_data['tag_size']

# Initialize detector
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# Load image
gray = cv2.imread('test_image.png', cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise Exception("Image not found")

# Detect tags with pose estimate
tags = at_detector.detect(
    gray,
    estimate_tag_pose=True,
    camera_params=camera_params,
    tag_size=tag_size
)

# Print results (ID, position and rotation)
for tag in tags:
    print(f"Tag ID: {tag.tag_id}")
    print(f"Position (t): {tag.pose_t}")
    print(f"Rotation (R): \n{tag.pose_R}")

# Visualisation - draw bounding boxes around tags and identify with ID
colour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

for tag in tags:
    corners = tag.corners.astype(int)

    # Draw box
    for i in range(4):
        cv2.line(colour_img,
                 tuple(corners[i - 1]),
                 tuple(corners[i]),
                 (0, 255, 0), 2)

    # Draw ID
    cv2.putText(colour_img,
                str(tag.tag_id),
                (corners[0][0] + 10, corners[0][1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2)

    # Draw centre
    centre = tuple(tag.center.astype(int))
    cv2.circle(colour_img, centre, 5, (255, 0, 0), -1)

# Display result in image
cv2.imshow("Detected Tags", colour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()