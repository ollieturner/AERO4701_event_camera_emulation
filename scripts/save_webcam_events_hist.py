#!/usr/bin/env python3

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
    # Create histogram output folder
    # ==============================
    hist_dir = "histograms"
    os.makedirs(hist_dir, exist_ok=True)
    frame_idx = 0

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
            # HISTOGRAM GENERATION
            # ==============================
            # Convert RGB event image → grayscale event magnitude
            gray_event = cv2.cvtColor(event_image, cv2.COLOR_BGR2GRAY)

            # Normalize for visualisation
            hist_vis = cv2.normalize(gray_event, None, 0, 255, cv2.NORM_MINMAX)

            # Save histogram image
            hist_filename = os.path.join(hist_dir, f"hist_{frame_idx:05d}.png")
            cv2.imwrite(hist_filename, hist_vis)

            # Optional: save raw histogram data
            # np.save(os.path.join(hist_dir, f"hist_{frame_idx:05d}.npy"), gray_event)

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
            cv2.imshow('Histogram', hist_vis)

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