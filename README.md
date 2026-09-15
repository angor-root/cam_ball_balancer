# cam_ball_balancer

Vision subsystem for the MT5001 (DSM) ball-on-plate robot: a differential-drive
mobile base carrying a 2-DOF tiltable plate that has to balance a ball.

This package covers the two things a single fixed overhead camera can see:

1. **Where the ball is** on the plate — color/shape detection + a Kalman
   filter, so the estimate keeps predicting sensibly even through short
   detection gaps (motion blur, an arm crossing the frame, etc.).
2. **How the plate is actually tilted** (θx, θy) — there is no encoder
   on the gimbal, so 4 ArUco tags mounted at the plate corners are the
   *only* source of truth for its orientation. This is not an optional
   extra; it's as central as ball tracking.

Both are exposed two ways: as plain Python (no ROS dependency — you can
develop and test this at a desk with a webcam, no robot needed) and as
ROS2 nodes/topics/TF, ready to be dropped into the rest of the robot's
ROS2 stack once it exists.

## Status (2026-09-15)

Repo moved out of `~/curso_mecatronica_pcb/` (a different course's
folder it was nested under) into its own top-level location and pushed
public: https://github.com/angor-root/cam_ball_balancer (MIT license).

**Design decision: state machine, not a neural network, for ball-detection
noise.** The real camera showed intermittent bad detections (lighting
glare/reflections briefly read as the ball). Considered a learned
detector instead of tuning the classical HSV pipeline further — decided
against it: no labeled real-condition dataset exists yet, RPi4 has no
inference accelerator, and a NN would still need the exact same
temporal-continuity logic on top of it (something has to decide "trust
this detection or coast on the prediction"), so it doesn't remove the
need for a state machine, only adds training cost. Implemented instead:
`BallTracker` now has an explicit `TrackState`
(SEARCHING/TRACKING/COASTING/LOST) with a kinematic outlier gate
(`gate_max_speed`/`gate_min_jump` — deliberately not a Mahalanobis gate
against the Kalman's own covariance, which underestimates real
per-frame residuals on curved motion and ends up rejecting good
detections; see `ball_tracker.py` docstrings). `platform_pose` (the tag
side) doesn't have an equivalent hold-last-valid layer yet — deferred
until the plate is physically assembled and tags are actually being
read, so it's tuned against real dropout behavior instead of guessed.

## Status (2026-09-14)

Real camera confirmed working end-to-end through ROS2 + RViz (`scripts/
publish_camera.py`, generic USB webcam at `/dev/video2` on the dev
machine, index may differ once wired into the actual PCB/RPi4). No
plate or printed tags yet — the mechanical platform design is still
pending (see the CAD work in `plataforma_equilibrio_3rps/`, not
assembled). `platform_pose`'s tag-based estimation is therefore still
only exercised against synthetic data.

Added today:
- **Salt-and-pepper noise fix**: a `cv2.medianBlur` pre-filter in
  `BallDetector.detect()` (new `BallDetectorConfig.median_kernel`,
  default 5), applied before the existing Gaussian blur. Median
  removes impulsive per-pixel noise that a Gaussian blur just smears
  around instead of removing.
- **`scripts/calibrate_camera.py`**: standard OpenCV chessboard
  calibration (interactive webcam capture or batch from a folder of
  photos). Saves `camera_matrix`/`dist_coeffs` directly in the `.npz`
  shape `CameraIntrinsics.load()` expects.
- **`scripts/publish_camera.py`**: real-camera equivalent of
  `publish_synthetic_video.py` — `cv2.VideoCapture` -> ROS2 Image
  topic. USB only; a CSI camera (Pi Camera Module) needs a different
  capture path (libcamera/picamera2), not this script.
- A calibration chessboard pattern (10x7 squares = 9x6 internal
  corners, matches both scripts' defaults) to display full-screen on a
  tablet instead of printing it — flatter than paper. **Measure one
  square with a ruler**; screen size varies per device so the mm size
  can't be assumed, it's a required `--square-size-mm` argument.

**Before trusting any of this on the real robot**, go through the
"When the hardware arrives" checklist near the bottom (now partially
done — camera capture path is proven, calibration is not).

## Layout

This repo *is* a colcon workspace (not just one package):

```
cam_ball_balancer/
├── src/
│   ├── cam_ball_balancer_msgs/     # BallPosition.msg, PlatformPose.msg (ament_cmake)
│   └── cam_ball_balancer/          # the actual package (ament_python)
│       ├── cam_ball_balancer/
│       │   ├── ball_tracker.py      # detector + Kalman, ROS-free
│       │   ├── kalman.py            # constant-velocity 2D Kalman filter
│       │   ├── platform_pose.py     # ArUco corner tags -> plate pose, ROS-free
│       │   ├── homography.py        # pixel <-> plate-plane (meters) mapping
│       │   ├── synthetic.py         # test data generators (no hardware needed)
│       │   ├── ball_tracker_node.py     # ROS2 wrapper
│       │   └── platform_pose_node.py    # ROS2 wrapper
│       ├── launch/cam_ball_balancer.launch.py
│       ├── config/{params.yaml, rviz.rviz}
│       └── test/             # pytest, all ROS-free
└── scripts/publish_synthetic_video.py   # feeds fake frames over ROS2 for a full demo
```

## Quickstart — no ROS, no camera (core algorithm only)

```bash
cd src/cam_ball_balancer
python3 -m pip install --user opencv-python numpy pytest   # or use your distro's python3-opencv
PYTHONPATH=. python3 -m pytest test/ -v
```

16 tests, covering: ball tracking accuracy on a moving target, tracking
*through* dropped frames, track-loss after a long occlusion, the
homography math, and platform pose recovery against known ground-truth
tilt angles (via a perspective-correct synthetic renderer — see
`TestPerspectiveCorrectRendering` in `test_platform_pose.py`).

## Quickstart — with ROS2

Built and tested against **ROS2 Jazzy on Ubuntu 24.04** (this dev
machine). The team's original architecture plan targeted Humble/22.04
for the RPi4 — the code only uses standard rclpy/cv_bridge APIs so it
should work unmodified on Humble, but re-run the test suite there
before trusting it.

```bash
source /opt/ros/jazzy/setup.bash   # or humble, on the RPi4
colcon build --symlink-install
source install/setup.bash

# Full demo with zero hardware: synthetic ball+plate video over ROS2
python3 scripts/publish_synthetic_video.py &
ros2 launch cam_ball_balancer cam_ball_balancer.launch.py image_topic:=/synthetic/image_raw
```

RViz should open with the debug camera view, the ball as an orange
sphere, and the plate's TF frame (`plate_link`) tilting under
`camera_link`. With a real camera, just point `image_topic` at it
instead (e.g. `usb_cam`'s `/image_raw`, or a CSI driver's topic).

### Topics (all relative — see docstrings)

| Topic | Type | From |
|---|---|---|
| `ball_position` | `cam_ball_balancer_msgs/BallPosition` (x, y, t via header, valid) | ball_tracker_node |
| `ball_marker` | `visualization_msgs/Marker` | ball_tracker_node |
| `debug_image` | `sensor_msgs/Image` | ball_tracker_node |
| `platform_pose` | `cam_ball_balancer_msgs/PlatformPose` (θx, θy, tx/ty/tz, valid) | platform_pose_node |
| `platform_pose_stamped` | `geometry_msgs/PoseStamped` | platform_pose_node |
| `tf`: `camera_link -> plate_link` | | platform_pose_node |

Both nodes use **relative** topic names on purpose. Run them under a
launch-time namespace (e.g. `namespace="cam_ball_balancer"`) once they're
folded into the full robot's launch files, and everything above gets
prefixed automatically — no code changes needed here. That's the
integration hook for the differential-drive base later.

## Design notes / why things are built this way

- **Ball tracking uses a constant-velocity Kalman filter, tuned for low
  lag over smooth velocity** (`kalman.py`'s `KalmanConfig` defaults).
  The ball's motion on a tilting plate is never truly constant-velocity,
  so there's an inherent trade-off between filter lag on turns and
  noise rejection; the defaults favor position accuracy (what the
  interface actually returns) over a clean velocity estimate (which
  isn't exposed). Re-tune `process_noise_vel`/`measurement_noise`
  against real footage once available — `test_ball_tracker.py`
  documents the trade-off with numbers.

- **Platform pose uses `cv2.SOLVEPNP_IPPE`**, not plain
  `SOLVEPNP_ITERATIVE`. A near-fronto-parallel planar target — exactly
  our case, overhead camera looking at a mostly-flat plate — has a
  well-known pose ambiguity: two rotations (the true one and a ~180°
  "ghost" flip) fit the 2D projection almost equally well. IPPE is the
  estimator OpenCV ships specifically for this; it returns both
  candidates ranked by reprojection error and we take the best one.
  This is a real bug that hit initial development (see git history /
  the corner-order comments in `platform_pose.py`) — the regression
  tests in `test_platform_pose.py` (`TestPerspectiveCorrectRendering`)
  exist specifically to catch it coming back.

- **`platform_pose.py`'s plate frame is image-aligned**: Y grows the
  same direction as the image row/v coordinate (not "up" in the
  intuitive sense). This has to stay consistent between
  `corner_positions_m` (where you say each tag is mounted) and the
  per-marker corner template used internally — getting it backwards
  silently flips the recovered tilt by ~180° about the plate's own X
  axis. Read the comments in `PlatformPoseConfig` before changing the
  corner layout.

- **Both OpenCV ArUco APIs are supported** (`_ArucoCompat` in
  `platform_pose.py`): the old free-function API (opencv-python <4.7,
  what this dev machine has: 4.6) and the new `cv2.aruco.ArucoDetector`
  class API (>=4.7). The RPi4's OpenCV version is unknown — this should
  work either way, but only the old-API path has actually been
  exercised (see Status above).

## Configuration

`config/params.yaml` — HSV thresholds for the ball, marker IDs/layout
for the plate corners, Kalman/detector tuning. Nothing in it has been
validated against real hardware; it's a starting point.

## When the hardware arrives — checklist

1. **Calibrate the camera** (intrinsics + distortion) — DONE
   (2026-09-14): `scripts/calibrate_camera.py` (see above), RMS
   reprojection error 0.28px. `platform_pose_node.py` now auto-loads
   `config/camera_intrinsics.npz` on startup (new `camera_intrinsics_path`
   parameter, defaults to that file inside the installed package share
   dir — falls back to `identity_guess()` with a warning if it's
   missing). **Re-run `colcon build` any time you re-calibrate** — the
   `.npz` only reaches `install/` through the same `data_files` glob
   that already installs `params.yaml`/`rviz.rviz` (see `setup.py`), so
   a stale build silently keeps serving the old calibration.
2. **Tune `hsv_lower`/`hsv_upper`** in `config/params.yaml` against the
   real ball under real lighting (`BallDetector.detect()`'s mask step —
   an interactive `cv2.createTrackbar` script would help here, not
   built yet).
3. **Measure and set `corner_positions_m`, `marker_length_m`** for the
   real plate (mind the image-aligned convention above).
4. **Re-run the whole test suite**, then validate against a short real
   video before trusting either node's output in the control loop.
5. Persona C's original risk note still applies: test ball detection
   with the camera under simulated vibration, not just on a static
   desk — the base moves.

## License

MIT.
