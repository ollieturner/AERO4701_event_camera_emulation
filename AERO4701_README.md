(put in setup instructions)
(make clean and proper)

Instructions on running code:

* Setup/activate venv:
  source venv/bin/activate
* Run custom script:
  python3 scripts/custom_stream_camera_events_w_video.py

Will save videos into a folder called videos
videos/ is untracked by .gitignore - upload these separately to google drive to avoid pushing videos to git



run frame_tag_calibration to calibrate camera intrinsics

then frame_tag_pose to use on (later live) video stream