import cv2
from dt_apriltags import Detector

# 1. Initialize detector
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# 2. Load image
gray = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

if gray is None:
    raise Exception("Image not found")

# 3. Detect tags (no pose needed for drawing)
tags = at_detector.detect(gray, estimate_tag_pose=False)

# 4. Print results
for tag in tags:
    print(f"Tag ID: {tag.tag_id}")
    print(f"Center: {tag.center}")
    print(f"Corners: {tag.corners}")

# 5. Convert to color for drawing
color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# 6. Draw detections
for tag in tags:
    corners = tag.corners.astype(int)

    # Draw bounding box
    for i in range(4):
        pt1 = tuple(corners[i - 1])
        pt2 = tuple(corners[i])
        cv2.line(color_img, pt1, pt2, (0, 255, 0), 2)

    # Draw tag ID
    cv2.putText(color_img,
                str(tag.tag_id),
                (corners[0][0] + 10, corners[0][1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2)

    # Optional: draw center
    center = tuple(tag.center.astype(int))
    cv2.circle(color_img, center, 5, (255, 0, 0), -1)

# 7. Show image
cv2.imshow("Detected Tags", color_img)
cv2.waitKey(0)
cv2.destroyAllWindows()