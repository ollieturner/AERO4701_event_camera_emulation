import cv2
from dt_apriltags import Detector


# 1. Initialize the detector
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


# MAKE SURE IN RIGHT FOLDER/ADD FOLDER PATH
gray = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# 3. Detect tags
# tags = at_detector.detect(gray)
tags = at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)

# 4. Access results
for tag in tags:
    print(f"Tag ID: {tag.tag_id}")
    print(f"Center: {tag.center}")
    print(f"Corners: {tag.corners}")