from picamera2 import Picamera2
import cv2
import os

picam2 = Picamera2()

save_dir = "frames"
os.makedirs(save_dir, exist_ok=True)

config = picam2.create_preview_configuration()
picam2.configure(config)

picam2.start()

frame_id = 0

try:
    while True:
        frame = picam2.capture_array()

        # SHOW LIVE STREAM
        cv2.imshow("Camera", frame)

        # SAVE FRAME
        filename = f"{save_dir}/frame_{frame_id:06d}.jpg"
        cv2.imwrite(filename, frame)
        frame_id += 1

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped")

finally:
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()




# ~ from picamera2 import Picamera2
# ~ import time
# ~ import os

# ~ picam2 = Picamera2()

# ~ save_dir = "frames"
# ~ os.makedirs(save_dir, exist_ok=True)

# ~ config = picam2.create_preview_configuration()
# ~ picam2.configure(config)

# ~ picam2.start()

# ~ FRAME_LIMIT = None   # or None for infinite
# ~ frame_id = 0

# ~ try:
    # ~ while True:
        # ~ frame = picam2.capture_array()

        # ~ filename = f"{save_dir}/frame_{frame_id:06d}.jpg"
        # ~ picam2.capture_file(filename)

        # ~ frame_id += 1

        # ~ if FRAME_LIMIT is not None and frame_id >= FRAME_LIMIT:
            # ~ break

# ~ except KeyboardInterrupt:
    # ~ print("Stopped")

# ~ finally:
    # ~ picam2.stop()
    # ~ picam2.close()
