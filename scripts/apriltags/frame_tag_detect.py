# Baseline/simple tag detection in a single image read from file
# Prints out tag IDs, centre and corners

# Make sure in apriltags/ folder in order to read image.png properly

import cv2
from dt_apriltags import Detector

# Initialise the detector
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# Read image and turn into grayscale for tag detection
gray = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Detect tags
tags = at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)

# Print out ID, centre and corner coordinates for each tag detected
for tag in tags:
    print(f"Tag ID: {tag.tag_id}")
    print(f"Centre: {tag.center}")
    print(f"Corners: {tag.corners}")