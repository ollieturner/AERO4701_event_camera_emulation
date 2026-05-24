# ORIGINAL, TUNED
import cv2
from dt_apriltags import Detector
import numpy as np

# Initialise detector
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# # Tuned a bit
# at_detector = Detector(families='tag36h11',
#                        nthreads=1,
#                        quad_decimate=2.0,
#                        quad_sigma=0.5,
#                        refine_edges=1,
#                        decode_sharpening=0.1,
#                        debug=0)

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture('event.mp4')

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
out = cv2.VideoWriter('event_with_tags.mp4', fourcc, fps, (w, h))

print("Processing and saving video...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blurring:
        # gray = cv2.GaussianBlur(gray, (9, 9), 0)

        # # Optional: boost contrast
        # gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


        # # Threshold attempt
        # # Keep only white pixels
        # # Threshold for "white"
        # lower = (200, 200, 200)
        # upper = (255, 255, 255)

        # mask = cv2.inRange(frame, lower, upper)

        # # Convert to clean binary image
        # gray = mask  # already 0 or 255

        # # gray = cv2.GaussianBlur(gray, (7, 7), 0)



        # # Next attempt - filling
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # Fill regions
        # gray = cv2.GaussianBlur(gray, (15, 15), 0)

        # # Force binary
        # _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)



        # # Morphology
        # # Convert to grayscale
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # Binary image (edges only)
        # _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # gray = cv2.GaussianBlur(gray, (15, 15), 0)

        # # -----------------------------
        # # Morphological operations
        # # -----------------------------
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # # 1. Dilate → thicken lines
        # gray = cv2.dilate(gray, kernel, iterations=2)

        # # 2. Close → fill small gaps
        # gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)




        # # Flood fill 
        # lower = (200, 200, 200)
        # upper = (255, 255, 255)

        # mask = cv2.inRange(frame, lower, upper)

        # # -----------------------------
        # # Flood fill to clean regions
        # # -----------------------------

        # # Make a copy (floodFill modifies image in-place)
        # flood = mask.copy()

        # h, w = flood.shape[:2]
        # flood_mask = np.zeros((h + 2, w + 2), np.uint8)

        # # Pick a seed point (top-left corner is common if background is connected)
        # seed_point = (0, 0)

        # # Flood fill background to 0 (black)
        # cv2.floodFill(flood, flood_mask, seed_point, 0)

        # # Invert flood-filled image to get foreground mask
        # flood_filled_inv = cv2.bitwise_not(flood)

        # # Combine original mask with flood-filled result
        # clean_mask = cv2.bitwise_or(mask, flood_filled_inv)

        # # Final binary image
        # gray = clean_mask





        # # Flood fill 2
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # Step 1: binary edge image (your detected outlines)
        # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # # Step 2: invert (so interiors become flood-fill targets)
        # inv = cv2.bitwise_not(binary)

        # # Step 3: flood fill from image borders to remove background
        # h, w = inv.shape
        # ff_mask = np.zeros((h + 2, w + 2), np.uint8)

        # cv2.floodFill(inv, ff_mask, (0, 0), 0)

        # # Step 4: invert back → now enclosed regions are filled
        # filled = cv2.bitwise_not(inv)

        # # Step 5: combine with original edges if needed
        # gray = cv2.bitwise_or(binary, filled)



        # Flood fill 3
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # binary edges
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # CLOSE gaps so regions become enclosed
        kernel = np.ones((5,5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        h, w = closed.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)

        ff = closed.copy()

        cv2.floodFill(ff, mask, (0, 0), 0)  # remove background

        filled = cv2.bitwise_not(ff)

        # filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))




        # Checkerboard then PnP
        # To use edges 

        # Least squares on three checkerboards 


        # Detect tags
        tags = at_detector.detect(gray, estimate_tag_pose=False)

        # Draw detections
        colour_img = frame.copy()

        for tag in tags:
            print("Tag detected")
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
        # out.write(colour_img)          
        out.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))


except KeyboardInterrupt:
    print("\nInterrupted")

finally:
    cap.release()
    out.release()
    print("Saved video")
