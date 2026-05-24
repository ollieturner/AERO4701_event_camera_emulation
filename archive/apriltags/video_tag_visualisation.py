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

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture('original.mp4')

if not cap.isOpened():
    raise Exception("Could not open video")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -----------------------------
# Setup VideoWriter
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('original_with_tags.mp4', fourcc, fps, (w, h))

print("Processing and saving video...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect tags
        tags = at_detector.detect(gray, estimate_tag_pose=False)

        # Draw detections
        colour_img = frame.copy()

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

            # Draw centre
            centre = tuple(tag.center.astype(int))
            cv2.circle(colour_img, centre, 5, (255, 0, 0), -1)

        # Write frame to output video
        out.write(colour_img)

except KeyboardInterrupt:
    print("\nInterrupted")

finally:
    cap.release()
    out.release()
    print("Saved video")