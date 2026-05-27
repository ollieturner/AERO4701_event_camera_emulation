"""
experiment_io.py

Writes a single binary experiment_results.bin with the layout:

  [HISTOGRAMS]   Up to 150 rows × 38,400 bytes  (640×480×1 bit packed)
  [POSE]         Up to 150 rows × 12 bytes each
                   position row : x, y, z        (3× float32 = 12 bytes)
                   attitude row : roll, pitch, yaw (3× float32 = 12 bytes)

The number of histogram rows and pose rows are written as 4-byte
little-endian uint32 values at the very start of the file so a reader
knows how many rows to expect:

  Offset 0  : uint32  num_hist_rows
  Offset 4  : uint32  num_pose_rows
  Offset 8  : histogram data  (num_hist_rows × 38,400 bytes)
  Offset 8 + num_hist_rows*38400 : pose data (num_pose_rows × 24 bytes)
    each pose row = [x, y, z, roll, pitch, yaw] as 6× float32
"""

import os
import glob
import shutil
import time
import struct
import numpy as np
import cv2 as cv

HIST_ROW_BYTES = 38_400          # 640 × 480 × 1 bit / 8
POSE_ROW_BYTES = 24              # 6 × float32


# ---------------------------------------------------------------------------
# Writing helpers
# ---------------------------------------------------------------------------

def _pack_histogram(event_hist: np.ndarray) -> bytes:
    """Convert a float32 accumulation array to a 38,400-byte packed-bit row."""
    hist_binary = (event_hist > 0).astype(np.uint8)
    packed = np.packbits(hist_binary, axis=None)
    # Ensure exactly HIST_ROW_BYTES (pad or truncate for safety)
    if packed.nbytes < HIST_ROW_BYTES:
        packed = np.pad(packed, (0, HIST_ROW_BYTES - packed.nbytes))
    return packed.tobytes()[:HIST_ROW_BYTES]


def _pack_pose(x, y, z, roll, pitch, yaw) -> bytes:
    """Pack 6 floats into 24 bytes (little-endian float32)."""
    return struct.pack("<6f", x, y, z, roll, pitch, yaw)


# ---------------------------------------------------------------------------
# Record experiment video and save frames
# ---------------------------------------------------------------------------

def save_exp_video(picam2_, display_widget=False, save_debug_images=False,
                   exp_time=20.0, WINDOW_SIZE=1):
    print("Starting experiment")

    baseline_folder = "outputs/baseline"
    results_dir     = "outputs/experiment_results"
    shutil.rmtree(baseline_folder, ignore_errors=True)
    shutil.rmtree(results_dir, ignore_errors=True)
    os.makedirs(baseline_folder, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if save_debug_images:
        event_frame_dir = "outputs/event_data/frames"
        event_hist_dir  = "outputs/event_data/histograms"
        shutil.rmtree("outputs/event_data", ignore_errors=True)
        os.makedirs(event_frame_dir, exist_ok=True)
        os.makedirs(event_hist_dir, exist_ok=True)

    e_camera_emulator = EventCameraEmulator()
    frames     = []
    frame_idx  = 0
    hist_rows  = []          # list of bytes, one entry per completed window

    # Wait up to 10 s for first frame
    timeout    = 10.0
    start_time = time.time()
    frame      = None
    while frame is None and (time.time() - start_time) < timeout:
        try:
            frame = picam2_.capture_array("main")
        except Exception:
            frame = None
    if frame is None:
        print("[save_exp_video] [ERROR] No frame received within 10 s")
        return None
    prev_frame = frame
    print("[save_exp_video] [INFO] Read first frame")

    def process_frame(frame, prev_frame, frame_idx, event_hist):
        frames.append(frame)

        event_image = e_camera_emulator.get_events_image_rgb(
            frame, prev_frame, 30,
            record_off_events=True,
            register_off_events_as_on=False
        )
        visual_event_image = e_camera_emulator.get_visual_events_image(event_image)

        gray_event = (cv.cvtColor(event_image, cv.COLOR_BGR2GRAY)
                      if event_image.ndim == 3 else event_image).astype(np.float32)

        if event_hist is None:
            event_hist = np.zeros_like(gray_event, dtype=np.float32)
        event_hist += np.abs(gray_event)

        if frame_idx % WINDOW_SIZE == 0 and frame_idx > 0:
            hist_idx = frame_idx // WINDOW_SIZE
            if hist_idx < 150:
                hist_rows.append(_pack_histogram(event_hist))

        if save_debug_images:
            cv.imwrite(
                os.path.join(event_frame_dir, f"event_{frame_idx:04d}.png"),
                visual_event_image
            )
            if frame_idx % WINDOW_SIZE == 0 and frame_idx > 0:
                hist_vis = cv.normalize(event_hist, None, 0, 255,
                                        cv.NORM_MINMAX).astype(np.uint8)
                cv.imwrite(
                    os.path.join(event_hist_dir,
                                 f"event_hist_vis_{frame_idx:05d}.png"),
                    hist_vis
                )

        if frame_idx % WINDOW_SIZE == 0 and frame_idx > 0:
            event_hist = np.zeros_like(gray_event, dtype=np.float32)

        return event_hist

    start_time = time.time()
    event_hist = None

    if display_widget:
        try:
            while time.time() - start_time < exp_time:
                frame = picam2_.capture_array("main")
                if frame is None:
                    continue
                cv.imshow("Experiment Camera", frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Early exit triggered")
                    break
                event_hist = process_frame(frame, prev_frame, frame_idx, event_hist)
                prev_frame = frame
                frame_idx += 1
        except Exception as exc:
            print(f"[save_exp_video] [ERROR] Widget failed: {exc}")
            return None
        finally:
            cv.destroyAllWindows()
    else:
        while time.time() - start_time < exp_time:
            frame = picam2_.capture_array("main")
            if frame is None:
                continue
            event_hist = process_frame(frame, prev_frame, frame_idx, event_hist)
            prev_frame = frame
            frame_idx += 1

    if picam2_ is not None:
        close_camera(picam2_)

    for i, f in enumerate(frames):
        cv.imwrite(f"{baseline_folder}/frame_{i:04d}.jpeg", f)

    # Write histogram section of the binary file.
    # Pose rows are appended later by process_baseline_data().
    results_path = os.path.join(results_dir, "experiment_results.bin")
    num_hist = len(hist_rows)
    with open(results_path, "wb") as f:
        f.write(struct.pack("<I", num_hist))   # placeholder; pose count = 0 for now
        f.write(struct.pack("<I", 0))          # pose row count filled in later
        for row in hist_rows:
            f.write(row)

    print(f"Experiment recording complete — {num_hist} histogram rows written\n")
    return True


# ---------------------------------------------------------------------------
# Process baseline frames and append pose estimates to the binary file
# ---------------------------------------------------------------------------

def process_baseline_data(objpoints_3boards, mtx, dist, ROIS, CHESSBOARD=(5, 3),
                           baseline_folder="outputs/baseline",
                           pose_folder="outputs/baseline_pose",
                           save_debug_images=False):
    print("Processing baseline frames")

    images = sorted(glob.glob(os.path.join(baseline_folder, "*.jpeg")))
    print(f"Found images: {len(images)}")

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    os.makedirs(pose_folder, exist_ok=True)

    results_path = "outputs/experiment_results/experiment_results.bin"
    pose_rows = []

    for fname in images:
        img  = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        base = os.path.splitext(os.path.basename(fname))[0]

        for board_id, roi in enumerate(ROIS):
            x1, y1, x2, y2 = roi
            mask = np.zeros_like(gray, dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            working_img = cv.bitwise_and(gray, gray, mask=mask)

            ret, corners = cv.findChessboardCorners(working_img, CHESSBOARD, None)
            if not ret:
                pose_rows.append(_pack_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                continue

            corners = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            success, rvec, tvec = cv.solvePnP(
                objpoints_3boards[board_id], corners, mtx, dist
            )

            if success:
                t = tvec.flatten()
                r = rvec.flatten()
                pose_rows.append(_pack_pose(t[0], t[1], t[2],
                                            r[0], r[1], r[2]))
            else:
                pose_rows.append(_pack_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

            if save_debug_images and success:
                cv.drawChessboardCorners(img, CHESSBOARD, corners, ret)

        if save_debug_images:
            cv.imwrite(os.path.join(pose_folder, f"{base}_multi_pose.png"), img)

    # Append pose rows and patch the pose count in the file header
    num_pose = len(pose_rows)
    with open(results_path, "r+b") as f:
        f.seek(4)                              # byte 4 = pose row count field
        f.write(struct.pack("<I", num_pose))
        f.seek(0, 2)                           # jump to end of file
        for row in pose_rows:
            f.write(row)

    print(f"Baseline processing complete — {num_pose} pose rows written\n")
