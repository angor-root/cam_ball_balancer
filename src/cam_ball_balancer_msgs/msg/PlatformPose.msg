# Plate tilt, estimated from the 4 corner ArUco tags (no encoder exists
# for this angle — this topic IS the ground truth for plate orientation).
std_msgs/Header header

# Tilt about the plate's own X and Y axes, radians. Same (theta_x,
# theta_y) pair the ESP32 gimbal command already uses.
float64 theta_x
float64 theta_y

# Plate center position in the camera frame, meters (mainly for
# debugging / RViz; the control loop only needs theta_x, theta_y).
float64 tx
float64 ty
float64 tz

# False when fewer than 2 corner tags are visible (occlusion, tag
# damage, etc.) — consumers should not trust theta_x/theta_y.
bool valid
