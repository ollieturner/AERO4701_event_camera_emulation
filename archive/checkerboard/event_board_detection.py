import numpy as np
import cv2 as cv
from checkerboard import detect_checkerboard

# -----------------------------
# Settings
# -----------------------------
size = (9, 6)  # inner corners (IMPORTANT)

cap = cv.VideoCapture("event.mp4")

if not cap.isOpened():
    raise Exception("Could not open video")

fps = cap.get(cv.CAP_PROP_FPS)
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter("event_checkerboard_detected.mp4", fourcc, fps, (w, h))

print("Processing video...")

# -----------------------------
# Process frames
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Finished processing")
        break

    raw_frame = frame.copy()

    # -----------------------------
    # Convert event frame → grayscale signal
    # (adjusted for red/blue polarity)
    # -----------------------------
    frame_f = frame.astype(np.float32)

    event_img = frame_f[:, :, 2] - frame_f[:, :, 0]  # red - blue
    gray = cv.normalize(event_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # -----------------------------
    # Optional stabilisation (IMPORTANT for event data)
    # -----------------------------
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    # -----------------------------
    # Detect checkerboard
    # -----------------------------
    corners, score = detect_checkerboard(gray, size)

    if corners is not None:
        print("Detected checkerboard | score:", score)

        corners = np.array(corners, dtype=np.float32)

        # draw corners
        for c in corners:
            x, y = int(c[0]), int(c[1])
            cv.circle(raw_frame, (x, y), 3, (0, 255, 0), -1)

    # -----------------------------
    # Write output frame
    # -----------------------------
    out.write(raw_frame)

    # optional preview
    cv.imshow("detect", raw_frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()
cv.destroyAllWindows()


# import numpy as np
# import cv2 as cv
# from checkerboard import detect_checkerboard
# import glob

# size = (10, 7)  

# images = glob.glob('checkerboard_imgs/*.jpg')

# for fname in images:

#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     corners, score = detect_checkerboard(gray, size)

#     print("score:", score)

#     if corners is not None:

#         corners = np.array(corners, dtype=np.float32)

#         # optional visualization
#         for c in corners:
#             x, y = int(c[0]), int(c[1])
#             cv.circle(img, (x, y), 3, (0, 255, 0), -1)

#         cv.imshow("detected", img)
#         cv.waitKey(300)

# cv.destroyAllWindows()



# from checkerboard import detect_checkerboard
# import glob
# import cv2 as cv

# size = (10, 7) # size of checkerboard


# images = glob.glob('checkerboard_imgs/*.jpg')

# # cv.namedWindow('img', cv.WINDOW_NORMAL)
# # cv.resizeWindow('img', 600, 400)

# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     image = gray # obtain checkerboard

#     corners, score = detect_checkerboard(image, size)


#     print(score)

