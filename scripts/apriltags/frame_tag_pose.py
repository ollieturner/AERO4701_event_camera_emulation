# Detect tags, visualise in displayed image and extract pose estimate (position and rotation relative to camera)
# Using test_image.png and camera_info.yaml for pre-calibrated camera intrinsics 

import cv2
import numpy as np
import yaml
from dt_apriltags import Detector

# Load camera parameters from YAML
with open('camera_info.yaml', 'r') as f:
    cam_data = yaml.safe_load(f)

# with open('data.yaml', 'r') as f:
#     cam_data = yaml.safe_load(f)


# Extract K and reshape
K = np.array(cam_data['K']).reshape((3, 3))

# Extract intrinsics
fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

camera_params = (fx, fy, cx, cy)

# Tag size (in metres)
tag_size = cam_data['tag_size']

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