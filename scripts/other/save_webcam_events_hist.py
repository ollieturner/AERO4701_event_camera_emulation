#!/usr/bin/env python3

# source venv/bin/activate
# python3 save_webcam_events_hist.py


import sys
import argparse
import cv2
import numpy as np
import os

from event_camera_emulation.emulator import EventCameraEmulator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_device', '-v', type=str, default='0')
    args = parser.parse_args()

    camera_device = None
    original_writer = None
    event_writer = None

    # ==============================
    # Output folders
    # ==============================
    hist_dir = "event_histograms"
    os.makedirs(hist_dir, exist_ok=True)

    frame_idx = 0

    # ==============================
    # TRUE EVENT HISTOGRAM (spatial accumulation)
    # ==============================
    event_hist = None
    WINDOW_SIZE = 30  # reset accumulation every N frames (sliding window style)

    try:
        # -----------------------------
        # 1. Open webcam
        # -----------------------------
        try:
            camera_device = cv2.VideoCapture(int(args.video_device))
        except ValueError:
            camera_device = cv2.VideoCapture(args.video_device)

        if not camera_device.isOpened():
            print('[ERROR] Could not access camera')
            sys.exit()

        print('[INFO] Camera opened')

        ret, previous_image = camera_device.read()
        if not ret:
            print('[ERROR] Could not read first frame')
            sys.exit()

        # -----------------------------
        # 2. Event emulator
        # -----------------------------
        e_camera_emulator = EventCameraEmulator()

        # -----------------------------
        # 3. Video writer setup
        # -----------------------------
        fps = 30
        h, w = previous_image.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        original_writer = cv2.VideoWriter('original.mp4', fourcc, fps, (w, h))
        event_writer = cv2.VideoWriter('event.mp4', fourcc, fps, (w, h))

        # -----------------------------
        # 4. Main loop
        # -----------------------------
        while True:
            ret, current_image = camera_device.read()
            if not ret:
                break

            # -----------------------------------------
            # Event simulation from frame difference
            # -----------------------------------------
            event_image = e_camera_emulator.get_events_image_rgb(
                current_image,
                previous_image,
                30,
                record_off_events=True,
                register_off_events_as_on=False
            )

            visual_event_image = e_camera_emulator.get_visual_events_image(event_image)

            previous_image = current_image

            # ==============================
            # EVENT PROCESSING (FIXED)
            # ==============================
            if event_image.ndim == 3:
                gray_event = cv2.cvtColor(event_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_event = event_image

            gray_event = gray_event.astype(np.float32)

            # ==============================
            # TRUE SPATIAL EVENT HISTOGRAM
            # ==============================
            if event_hist is None:
                event_hist = np.zeros_like(gray_event, dtype=np.float32)

            event_hist += np.abs(gray_event)

            # -----------------------------
            # Optional sliding window reset
            # -----------------------------
            if frame_idx % WINDOW_SIZE == 0 and frame_idx > 0:
                event_hist /= WINDOW_SIZE  # smooth instead of full reset
                np.save(os.path.join(hist_dir, f"event_hist_{frame_idx:05d}.npy"), event_hist)

            # ==============================
            # VISUALISATION OF HISTOGRAM
            # ==============================
            hist_vis = cv2.normalize(event_hist, None, 0, 255, cv2.NORM_MINMAX)
            hist_vis = hist_vis.astype(np.uint8)

            # Save image version
            cv2.imwrite(
                os.path.join(hist_dir, f"hist_vis_{frame_idx:05d}.png"),
                hist_vis
            )

            # ==============================
            # OPTIONAL: true value histogram (distribution)
            # ==============================
            value_hist = np.bincount(
                np.clip(gray_event.ravel().astype(np.int32), 0, 255),
                minlength=256
            )

            np.save(
                os.path.join(hist_dir, f"value_hist_{frame_idx:05d}.npy"),
                value_hist
            )

            frame_idx += 1

            # ==============================
            # SAVE VIDEO
            # ==============================
            original_writer.write(current_image)
            event_writer.write(visual_event_image)

            # ==============================
            # DISPLAY
            # ==============================
            cv2.imshow('Original Camera stream', current_image)
            cv2.imshow('Simulated Event Camera stream', visual_event_image)
            cv2.imshow('Event Spatial Histogram', hist_vis)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    except KeyboardInterrupt:
        print('\n[INFO] Interrupted by user')

    finally:
        # -----------------------------
        # CLEANUP
        # -----------------------------
        print('[INFO] Releasing resources...')

        if camera_device is not None:
            camera_device.release()

        if original_writer is not None:
            original_writer.release()

        if event_writer is not None:
            event_writer.release()

        cv2.destroyAllWindows()




# #!/usr/bin/env python3

# # source venv/bin/activate
# # python3 save_webcam_events.py

# import sys
# import argparse
# import cv2
# import numpy as np
# import os

# from event_camera_emulation.emulator import EventCameraEmulator


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--video_device', '-v', type=str, default='0')
#     args = parser.parse_args()

#     camera_device = None
#     original_writer = None
#     event_writer = None

#     # ==============================
#     # Create histogram output folder
#     # ==============================
#     hist_dir = "histograms"
#     os.makedirs(hist_dir, exist_ok=True)
#     frame_idx = 0

#     try:
#         # -----------------------------
#         # 1. Open webcam
#         # -----------------------------
#         try:
#             camera_device = cv2.VideoCapture(int(args.video_device))
#         except ValueError:
#             camera_device = cv2.VideoCapture(args.video_device)

#         if not camera_device.isOpened():
#             print('[ERROR] Could not access camera')
#             sys.exit()

#         print('[INFO] Camera opened')

#         ret, previous_image = camera_device.read()
#         if not ret:
#             print('[ERROR] Could not read first frame')
#             sys.exit()

#         # -----------------------------
#         # 2. Event emulator
#         # -----------------------------
#         e_camera_emulator = EventCameraEmulator()

#         # -----------------------------
#         # 3. Video writer setup
#         # -----------------------------
#         fps = 30
#         h, w = previous_image.shape[:2]

#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#         original_writer = cv2.VideoWriter('original.mp4', fourcc, fps, (w, h))
#         event_writer = cv2.VideoWriter('event.mp4', fourcc, fps, (w, h))

#         # -----------------------------
#         # 4. Main loop
#         # -----------------------------
#         while True:
#             ret, current_image = camera_device.read()
#             if not ret:
#                 break

#             event_image = e_camera_emulator.get_events_image_rgb(
#                 current_image,
#                 previous_image,
#                 30,
#                 record_off_events=True,
#                 register_off_events_as_on=False
#             )

#             visual_event_image = e_camera_emulator.get_visual_events_image(event_image)

#             previous_image = current_image

#             # ==============================
#             # HISTOGRAM GENERATION
#             # ==============================
#             # Convert RGB event image → grayscale event magnitude
#             # gray_event = cv2.cvtColor(event_image, cv2.COLOR_BGR2GRAY)
#             if len(event_image.shape) == 3:
#                 gray_event = cv2.cvtColor(event_image, cv2.COLOR_BGR2GRAY)
#             else:
#                 gray_event = event_image

#             # Normalize for visualisation
#             hist_vis = cv2.normalize(gray_event, None, 0, 255, cv2.NORM_MINMAX)

#             # Save histogram image
#             hist_filename = os.path.join(hist_dir, f"hist_{frame_idx:05d}.png")
#             cv2.imwrite(hist_filename, hist_vis)

#             # Optional: save raw histogram data
#             # np.save(os.path.join(hist_dir, f"hist_{frame_idx:05d}.npy"), gray_event)

#             frame_idx += 1

#             # ==============================
#             # SAVE VIDEO
#             # ==============================
#             original_writer.write(current_image)
#             event_writer.write(visual_event_image)

#             # ==============================
#             # DISPLAY
#             # ==============================
#             cv2.imshow('Original Camera stream', current_image)
#             cv2.imshow('Simulated Event Camera stream', visual_event_image)
#             cv2.imshow('Histogram', hist_vis)

#             if cv2.waitKey(1) & 0xFF == 27:  # ESC
#                 break

#     except KeyboardInterrupt:
#         print('\n[INFO] Interrupted by user')

#     finally:
#         # -----------------------------
#         # CLEANUP
#         # -----------------------------
#         print('[INFO] Releasing resources...')

#         if camera_device is not None:
#             camera_device.release()

#         if original_writer is not None:
#             original_writer.release()

#         if event_writer is not None:
#             event_writer.release()

#         cv2.destroyAllWindows()