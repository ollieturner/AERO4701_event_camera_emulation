#!/usr/bin/env python3

import sys
import argparse
import cv2
import numpy as np
import time
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
    # Histogram setup
    # ==============================
    hist_dir = "event_histograms"
    os.makedirs(hist_dir, exist_ok=True)

    ACCUM_TIME = 0.2  # seconds
    last_flush_time = time.time()
    frame_idx = 0

    event_hist = None

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

            # -----------------------------
            # Event generation
            # -----------------------------
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
            # EVENT PROCESSING
            # ==============================
            if event_image.ndim == 3:
                gray_event = cv2.cvtColor(event_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_event = event_image

            gray_event = np.abs(gray_event).astype(np.float32)

            # ==============================
            # ACCUMULATE OVER TIME (0.2s)
            # ==============================
            if event_hist is None:
                event_hist = np.zeros_like(gray_event, dtype=np.float32)

            event_hist += gray_event

            current_time = time.time()

            if current_time - last_flush_time >= ACCUM_TIME:

                # -----------------------------
                # NORMALISE FOR VISUALISATION
                # -----------------------------
                hist_vis = cv2.normalize(event_hist, None, 0, 255, cv2.NORM_MINMAX)
                hist_vis = hist_vis.astype(np.uint8)

                # -----------------------------
                # SAVE HISTOGRAM IMAGE
                # -----------------------------
                cv2.imwrite(
                    os.path.join(hist_dir, f"hist_{frame_idx:05d}.png"),
                    hist_vis
                )

                # optional: save raw data
                np.save(
                    os.path.join(hist_dir, f"hist_{frame_idx:05d}.npy"),
                    event_hist
                )

                # RESET ACCUMULATOR
                event_hist[:] = 0
                last_flush_time = current_time

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
            cv2.imshow('Event Histogram (0.2s window)', hist_vis)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print('\n[INFO] Interrupted by user')

    finally:
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

# from event_camera_emulation.emulator import EventCameraEmulator


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--video_device', '-v', type=str, default='0')
#     args = parser.parse_args()

#     camera_device = None
#     original_writer = None
#     event_writer = None

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

#             # Save video
#             original_writer.write(current_image)
#             event_writer.write(visual_event_image)

#             # Display
#             cv2.imshow('Original Camera stream', current_image)
#             cv2.imshow('Simulated Event Camera stream', visual_event_image)

#             if cv2.waitKey(1) & 0xFF == 27:  # ESC
#                 break

#     except KeyboardInterrupt:
#         print('\n[INFO] Interrupted by user')

#     finally:
#         # -----------------------------
#         # 5. SAFE CLEANUP (ALWAYS RUNS)
#         # -----------------------------
#         print('[INFO] Releasing resources...')

#         if camera_device is not None:
#             camera_device.release()

#         if original_writer is not None:
#             original_writer.release()

#         if event_writer is not None:
#             event_writer.release()

#         cv2.destroyAllWindows()



# # #!/usr/bin/env python3

# # # source venv/bin/activate
# # # python3 save_webcam_events.py


# # import sys
# # import argparse
# # import cv2

# # from event_camera_emulation.emulator import EventCameraEmulator


# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--video_device', '-v', type=str, default='0')
# #     args = parser.parse_args()

# #     # -----------------------------
# #     # 1. Open webcam
# #     # -----------------------------
# #     try:
# #         camera_device = cv2.VideoCapture(int(args.video_device))
# #     except ValueError:
# #         camera_device = cv2.VideoCapture(args.video_device)

# #     if not camera_device.isOpened():
# #         print('[ERROR] Could not access camera')
# #         sys.exit()

# #     print('[INFO] Camera opened')

# #     ret, previous_image = camera_device.read()
# #     if not ret:
# #         print('[ERROR] Could not read first frame')
# #         sys.exit()

# #     # -----------------------------
# #     # 2. Event emulator
# #     # -----------------------------
# #     e_camera_emulator = EventCameraEmulator()

# #     # -----------------------------
# #     # 3. Video writer setup
# #     # -----------------------------
# #     fps = 30
# #     h, w = previous_image.shape[:2]

# #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# #     original_writer = cv2.VideoWriter('original.mp4', fourcc, fps, (w, h))
# #     event_writer = cv2.VideoWriter('event.mp4', fourcc, fps, (w, h))

# #     # -----------------------------
# #     # 4. Main loop
# #     # -----------------------------
# #     try:
# #         while True:
# #             ret, current_image = camera_device.read()
# #             if not ret:
# #                 break

# #             event_image = e_camera_emulator.get_events_image_rgb(
# #                 current_image,
# #                 previous_image,
# #                 30,
# #                 record_off_events=True,
# #                 register_off_events_as_on=False
# #             )

# #             visual_event_image = e_camera_emulator.get_visual_events_image(event_image)

# #             previous_image = current_image

# #             # -----------------------------
# #             # Save video
# #             # -----------------------------
# #             original_writer.write(current_image)
# #             event_writer.write(visual_event_image)

# #             # -----------------------------
# #             # Display
# #             # -----------------------------
# #             cv2.imshow('Original Camera stream', current_image)
# #             cv2.imshow('Simulated Event Camera stream', visual_event_image)

# #             if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
# #                 break

# #     except KeyboardInterrupt:
# #         print('\n[INFO] Exiting...')

# #     # -----------------------------
# #     # 5. Cleanup
# #     # -----------------------------
# #     camera_device.release()
# #     original_writer.release()
# #     event_writer.release()
# #     cv2.destroyAllWindows()