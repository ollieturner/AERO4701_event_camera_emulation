#!/usr/bin/env python3

"""Raspberry Pi Camera Module 3 event-emulation stream with recording.

Saves:
    - Original camera stream → test/original.mp4
    - Event camera stream → test/event.mp4
    - Runs for 10 seconds automatically
"""

import sys
import time
import argparse
import os

import cv2
from picamera2 import Picamera2
from libcamera import controls

from event_camera_emulation.emulator import EventCameraEmulator


picam2_ = None


def get_valid_frame(camera):
    frame = camera.capture_array("main")
    if frame is None or frame.size == 0:
        return None

    if len(frame.shape) != 3:
        return None

    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.shape[2] != 3:
        return None

    return frame


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=50.0)
    parser.add_argument('--threshold', type=float, default=30.0)
    parser.add_argument('--warmup', type=float, default=1.0)
    parser.add_argument('--exposure-us', type=int, default=None)
    parser.add_argument('--gain', type=float, default=None)
    parser.add_argument('--grayscale', action='store_true')
    args = parser.parse_args()

    frame_us = int(1_000_000 / args.fps)

    # ---------------- CAMERA SETUP ----------------
    try:
        picam2_ = Picamera2()

        config = picam2_.create_video_configuration(
            main={"format": "BGR888", "size": (args.width, args.height)},
            controls={"FrameDurationLimits": (frame_us, frame_us)}
        )
        picam2_.configure(config)
        picam2_.start()

        time.sleep(args.warmup)

        try:
            picam2_.set_controls({"AfMode": controls.AfModeEnum.Auto})
            success = picam2_.autofocus_cycle()
            meta = picam2_.capture_metadata()
            lens_pos = meta.get("LensPosition", None)

            if success and lens_pos is not None:
                picam2_.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": lens_pos
                })
        except Exception:
            pass

        meta = picam2_.capture_metadata()
        settled_exposure = int(meta.get("ExposureTime", min(5000, frame_us // 2)))
        settled_gain = float(meta.get("AnalogueGain", 1.0))
        colour_gains = meta.get("ColourGains", None)

        if args.exposure_us is not None:
            settled_exposure = args.exposure_us
        else:
            settled_exposure = min(settled_exposure, max(1000, int(0.6 * frame_us)))

        if args.gain is not None:
            settled_gain = args.gain

        lock_controls = {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": settled_exposure,
            "AnalogueGain": settled_gain,
            "FrameDurationLimits": (frame_us, frame_us),
        }

        if colour_gains is not None:
            lock_controls["ColourGains"] = colour_gains

        picam2_.set_controls(lock_controls)
        time.sleep(0.2)

        previous_image = get_valid_frame(picam2_)
        if previous_image is None:
            print('[ERROR] No valid initial frame.', flush=True)
            sys.exit(1)

        if args.grayscale:
            previous_image = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
            previous_image = cv2.cvtColor(previous_image, cv2.COLOR_GRAY2BGR)

        print('[INFO] Camera started', flush=True)

    except Exception as exc:
        print(f'[ERROR] Camera init failed: {exc}', flush=True)
        sys.exit(1)

    # ---------------- VIDEO SETUP ----------------
    output_dir = "videos"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Saving videos to: {os.path.abspath(output_dir)}", flush=True)

    original_path = os.path.join(output_dir, "original.mp4")
    event_path = os.path.join(output_dir, "event.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    original_out = cv2.VideoWriter(
        original_path,
        fourcc,
        args.fps,
        (args.width, args.height)
    )

    event_out = cv2.VideoWriter(
        event_path,
        fourcc,
        args.fps,
        (args.width, args.height)
    )

    print("[INFO] Writing:", flush=True)
    print("  ", original_path, flush=True)
    print("  ", event_path, flush=True)

    print("[INFO] original writer open:", original_out.isOpened(), flush=True)
    print("[INFO] event writer open:", event_out.isOpened(), flush=True)

    # ---------------- RUNTIME CONTROL ----------------
    start_time = time.time()
    runtime = 30.0

    # ---------------- EMULATOR ----------------
    e_camera_emulator = EventCameraEmulator()

    try:
        while True:

            # ---- 10 second exit ----
            if time.time() - start_time > runtime:
                print("[INFO] 10 second runtime reached, exiting...", flush=True)
                break

            current_image = get_valid_frame(picam2_)
            if current_image is None:
                continue

            if args.grayscale:
                current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
                current_image = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)

            event_image = e_camera_emulator.get_events_image_rgb(
                current_image,
                previous_image,
                args.threshold,
                record_off_events=True,
                register_off_events_as_on=False
            )

            visual_event_image = e_camera_emulator.get_visual_events_image(event_image)

            previous_image = current_image

            # -------- Save frames --------
            original_out.write(current_image)
            event_out.write(visual_event_image)

            cv2.imshow('Original Camera stream', current_image)
            cv2.imshow('Event Camera stream', visual_event_image)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("[INFO] Exit key pressed", flush=True)
                break

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt", flush=True)

    finally:
        print("[INFO] Releasing video files...", flush=True)

        if picam2_ is not None:
            picam2_.stop()

        original_out.release()
        event_out.release()

        cv2.destroyAllWindows()

        print("[INFO] Done. Files saved in:", os.path.abspath(output_dir), flush=True)
