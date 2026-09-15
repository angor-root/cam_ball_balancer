"""ROS2 node: wraps PlatformPoseEstimator around a camera topic.

Publishes (relative names — see ball_tracker_node.py's docstring for
why, and how to namespace this for the full robot later):
  platform_pose          (dsm_vision_msgs/PlatformPose)
  platform_pose_stamped  (geometry_msgs/PoseStamped)  -- for RViz / other consumers
and broadcasts a TF transform camera_frame -> plate_frame, so the plate's
pose is available on the standard ROS2 TF tree — this is the hook meant
for later integration with the differential-drive robot (e.g. a
navigation/control node just does a TF lookup instead of subscribing to
this package's custom message).
"""
from __future__ import annotations

import os

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
from ament_index_python.packages import get_package_share_directory

from dsm_vision_msgs.msg import PlatformPose as PlatformPoseMsg

from .platform_pose import PlatformPoseEstimator, PlatformPoseConfig, CameraIntrinsics


def _default_intrinsics_path() -> str:
    """Ruta al .npz calibrado dentro del paquete instalado (share/dsm_vision/
    config/), el mismo config/ donde ya viven params.yaml y rviz.rviz (ver
    setup.py -- data_files hace glob("config/*"), por eso camera_intrinsics.npz
    solo aparece ahi despues de un `colcon build` que lo recoja).
    """
    try:
        share_dir = get_package_share_directory("dsm_vision")
    except Exception:
        return ""
    return os.path.join(share_dir, "config", "camera_intrinsics.npz")


class PlatformPoseNode(Node):
    def __init__(self):
        super().__init__("platform_pose_node")

        self.declare_parameter("image_topic", "image_raw")
        self.declare_parameter("plate_frame_id", "plate_link")
        self.declare_parameter("marker_length_m", 0.03)
        self.declare_parameter("corner_marker_ids", [0, 1, 2, 3])
        self.declare_parameter(
            "corner_positions_m",
            [-0.09, -0.09, 0.09, -0.09, 0.09, 0.09, -0.09, 0.09],  # flattened x0,y0,x1,y1,...
        )
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("camera_intrinsics_path", _default_intrinsics_path())

        flat = self.get_parameter("corner_positions_m").value
        positions = tuple((flat[i], flat[i + 1]) for i in range(0, len(flat), 2))

        config = PlatformPoseConfig(
            marker_length_m=self.get_parameter("marker_length_m").value,
            corner_marker_ids=tuple(self.get_parameter("corner_marker_ids").value),
            corner_positions_m=positions,
        )

        intrinsics_path = self.get_parameter("camera_intrinsics_path").value
        intrinsics = None
        if intrinsics_path and os.path.isfile(intrinsics_path):
            intrinsics = CameraIntrinsics.load(intrinsics_path)
            self.get_logger().info(f"Intrinsecos calibrados cargados de {intrinsics_path}")
        else:
            self.get_logger().warn(
                f"No se encontro {intrinsics_path or '(sin ruta)'} -- usando identity_guess() "
                "sin calibrar. Corre scripts/calibrate_camera.py y recompila (colcon build) "
                "para que se instale junto a params.yaml."
            )

        self.estimator = PlatformPoseEstimator(config=config, intrinsics=intrinsics)
        self.plate_frame_id = self.get_parameter("plate_frame_id").value
        self.publish_tf = self.get_parameter("publish_tf").value

        self.bridge = CvBridge()
        image_topic = self.get_parameter("image_topic").value
        self.sub = self.create_subscription(Image, image_topic, self._on_image, 10)
        self.pub_pose = self.create_publisher(PlatformPoseMsg, "platform_pose", 10)
        self.pub_pose_stamped = self.create_publisher(PoseStamped, "platform_pose_stamped", 10)
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)

        calibrated_note = (
            "using calibrated intrinsics" if intrinsics is not None
            else "NOTE: intrinsics not loaded (using an uncalibrated pinhole "
                 "guess) — theta_x/theta_y will be approximate until "
                 "scripts/calibrate_camera.py output is wired in"
        )
        self.get_logger().info(
            f"PlatformPoseNode subscribed to '{image_topic}', "
            f"publishing plate pose as frame '{self.plate_frame_id}'. {calibrated_note}."
        )

    def _on_image(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pose = self.estimator.estimate(frame, t)

        out = PlatformPoseMsg()
        out.header = msg.header
        out.theta_x = pose.theta_x
        out.theta_y = pose.theta_y
        out.tx, out.ty, out.tz = pose.tx, pose.ty, pose.tz
        out.valid = pose.valid
        self.pub_pose.publish(out)

        if not pose.valid:
            return

        import cv2
        R, _ = cv2.Rodrigues(pose.rvec)
        qw, qx, qy, qz = _quat_from_rotation_matrix(R)

        stamped = PoseStamped()
        stamped.header = msg.header
        stamped.pose.position.x = pose.tx
        stamped.pose.position.y = pose.ty
        stamped.pose.position.z = pose.tz
        stamped.pose.orientation.x = qx
        stamped.pose.orientation.y = qy
        stamped.pose.orientation.z = qz
        stamped.pose.orientation.w = qw
        self.pub_pose_stamped.publish(stamped)

        if self.publish_tf:
            tf_msg = TransformStamped()
            tf_msg.header = msg.header
            tf_msg.child_frame_id = self.plate_frame_id
            tf_msg.transform.translation.x = pose.tx
            tf_msg.transform.translation.y = pose.ty
            tf_msg.transform.translation.z = pose.tz
            tf_msg.transform.rotation.x = qx
            tf_msg.transform.rotation.y = qy
            tf_msg.transform.rotation.z = qz
            tf_msg.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf_msg)


def _quat_from_rotation_matrix(R):
    """Standard rotation-matrix -> quaternion (w, x, y, z), no extra deps."""
    import numpy as np
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(w), float(x), float(y), float(z)


def main(args=None):
    rclpy.init(args=args)
    node = PlatformPoseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
