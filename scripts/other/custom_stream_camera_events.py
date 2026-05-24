#!/usr/bin/env python3

"""Raspberry Pi Camera Module 3 event-emulation stream with more stable capture.

Usage:
    python3 stream_camera_events.py
    python3 stream_camera_events.py --width 640 --height 480 --fps 60 --exposure-us 5000
"""

import sys
import time
import argparse

import cv2
from picamera2 import Picamera2
from libcamera import controls

from event_camera_emulation.emulator import EventCameraEmulator


picam2_ = None


def get_valid_frame(camera):
    frame = camera.capture_array("main")
    if frame is None or frame.size == 0:
        return None

    # Ensure 3-channel BGR
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
    parser.add_argument('--fps', type=float, default=50.0,
                        help='Target frame rate. Try 50 or 60 if lighting allows.')
    parser.add_argument('--threshold', type=float, default=30.0)
    parser.add_argument('--warmup', type=float, default=1.0,
                        help='Seconds to allow AE/AWB/AF to settle before locking.')
    parser.add_argument('--exposure-us', type=int, default=None,
                        help='Manual exposure time in microseconds. Example: 5000')
    parser.add_argument('--gain', type=float, default=None,
                        help='Manual analogue gain. Example: 2.0')
    parser.add_argument('--grayscale', action='store_true',
                        help='Convert frames to grayscale before event emulation.')
    args = parser.parse_args()

    frame_us = int(1_000_000 / args.fps)

    try:
        picam2_ = Picamera2()

        config = picam2_.create_video_configuration(
            main={"format": "BGR888", "size": (args.width, args.height)},
            controls={"FrameDurationLimits": (frame_us, frame_us)}
        )
        picam2_.configure(config)
        picam2_.start()

        # Let the auto algorithms settle first.
        time.sleep(args.warmup)

        # Try autofocus once, then lock the focus position.
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
            # If autofocus helpers are unavailable, continue without failing.
            pass

        # Read settled metadata and lock camera state.
        meta = picam2_.capture_metadata()
        settled_exposure = int(meta.get("ExposureTime", min(5000, frame_us // 2)))
        settled_gain = float(meta.get("AnalogueGain", 1.0))
        colour_gains = meta.get("ColourGains", None)

        if args.exposure_us is not None:
            settled_exposure = args.exposure_us
        else:
            # Keep exposure conservative for motion/event work.
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
            print('[stream_camera_events] [ERROR] No valid initial frame returned.')
            sys.exit(1)

        if args.grayscale:
            previous_image = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
            previous_image = cv2.cvtColor(previous_image, cv2.COLOR_GRAY2BGR)

        print('[stream_camera_events] [INFO] Camera started')
        print(f'[stream_camera_events] [INFO] FPS: {args.fps:.1f}, frame time: {frame_us} us')
        print(f'[stream_camera_events] [INFO] Locked exposure: {settled_exposure} us')
        print(f'[stream_camera_events] [INFO] Locked gain: {settled_gain:.3f}')

    except Exception as exc:
        print(f'[stream_camera_events] [ERROR] Could not access Raspberry Pi camera: {exc}')
        sys.exit(1)

    e_camera_emulator = EventCameraEmulator()

    try:
        while True:
            current_image = get_valid_frame(picam2_)
            if current_image is None:
                print('[stream_camera_events] [WARN] Empty/invalid frame received, skipping...')
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

            cv2.imshow('Original Camera stream', current_image)
            cv2.imshow('Simulated Event Camera stream', visual_event_image)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    except KeyboardInterrupt:
        print('\n[stream_camera_events] [INFO] Finished streaming, exiting program...')

    finally:
        if picam2_ is not None:
            picam2_.stop()
        cv2.destroyAllWindows()
