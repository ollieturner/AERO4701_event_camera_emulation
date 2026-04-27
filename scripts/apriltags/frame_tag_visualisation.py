# Tag detection and visualisation from a single image read from file
# Prints out ID, centre and corner coordinates for each tag + identifies them in displayed image

# Extension of frame_tag_detect

import cv2
from dt_apriltags import Detector

# Initialise detector
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# Load image
gray = cv2.imread('3_tags_ss.png', cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise Exception("Image not found")

# Detect tags
tags = at_detector.detect(gray, estimate_tag_pose=False)

# Print results
for tag in tags:
    print(f"Tag ID: {tag.tag_id}")
    print(f"Centre: {tag.center}")
    print(f"Corners: {tag.corners}")


## VISUALISATION ##

# Convert to colour
colour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Draw detections - use corners to draw lines around tags
for tag in tags:
    corners = tag.corners.astype(int)

    # Draw bounding box
    for i in range(4):
        pt1 = tuple(corners[i - 1])
        pt2 = tuple(corners[i])
        cv2.line(colour_img, pt1, pt2, (0, 255, 0), 2)

    # Draw tag ID
    cv2.putText(colour_img,
                str(tag.tag_id),
                (corners[0][0] + 10, corners[0][1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2)

    # Extra: draw centre
    centre = tuple(tag.center.astype(int))
    cv2.circle(colour_img, centre, 5, (255, 0, 0), -1)

# Display image with detected tags
cv2.imshow("Detected Tags", colour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()