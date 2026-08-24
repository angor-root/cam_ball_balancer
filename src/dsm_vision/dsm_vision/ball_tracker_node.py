"""ROS2 node: wraps BallTracker around a camera topic.

Publishes (relative names — see below):
  ball_position   (dsm_vision_msgs/BallPosition)
  ball_marker     (visualization_msgs/Marker)   -- for RViz
  debug_image     (sensor_msgs/Image)           -- annotated, for rqt_image_view

All topic names here are relative on purpose: run standalone they show
up as /ball_position etc., but this is meant to be one node among
several on the eventual differential-drive robot — launch it with
`namespace="dsm_vision"` (or whatever the robot's launch file uses) and
every topic above gets prefixed automatically, no code changes needed.
The published Marker/debug image reuse whatever `frame_id` the incoming
camera image carries, so a future robot_state_publisher + static camera
transform just slots in.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge

from dsm_vision_msgs.msg import BallPosition

from .ball_tracker import BallTracker, BallDetectorConfig
from .kalman import KalmanConfig


class BallTrackerNode(Node):
    def __init__(self):
        super().__init__("ball_tracker_node")

        self.declare_parameter("image_topic", "image_raw")
        self.declare_parameter("hsv_lower", [5, 120, 120])
        self.declare_parameter("hsv_upper", [18, 255, 255])
        self.declare_parameter("min_radius_px", 4.0)
        self.declare_parameter("max_radius_px", 200.0)
        self.declare_parameter("max_consecutive_misses", 15)
        self.declare_parameter("publish_debug_image", True)

        detector_config = BallDetectorConfig(
            hsv_lower=tuple(self.get_parameter("hsv_lower").value),
            hsv_upper=tuple(self.get_parameter("hsv_upper").value),
            min_radius_px=self.get_parameter("min_radius_px").value,
            max_radius_px=self.get_parameter("max_radius_px").value,
        )
        self.tracker = BallTracker(
            detector_config=detector_config,
            kalman_config=KalmanConfig(),
            plane_mapper=None,  # pixel coords until a homography/calibration is loaded
            max_consecutive_misses=self.get_parameter("max_consecutive_misses").value,
        )
        self.publish_debug = self.get_parameter("publish_debug_image").value

        self.bridge = CvBridge()
        image_topic = self.get_parameter("image_topic").value
        self.sub = self.create_subscription(Image, image_topic, self._on_image, 10)
        self.pub_position = self.create_publisher(BallPosition, "ball_position", 10)
        self.pub_marker = self.create_publisher(Marker, "ball_marker", 10)
        if self.publish_debug:
            self.pub_debug = self.create_publisher(Image, "debug_image", 10)

        self.get_logger().info(f"BallTrackerNode subscribed to '{image_topic}'")

    def _on_image(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        result = self.tracker.update(frame, t)

        out = BallPosition()
        out.header = msg.header
        out.x = result.x
        out.y = result.y
        out.valid = result.valid
        self.pub_position.publish(out)

        marker = Marker()
        marker.header = msg.header
        marker.ns = "dsm_vision/ball"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD if result.valid else Marker.DELETE
        if result.valid:
            marker.pose.position.x = result.x
            marker.pose.position.y = result.y
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.03
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.55
            marker.color.b = 0.0
        self.pub_marker.publish(marker)

        if self.publish_debug:
            self._publish_debug_image(frame, result, msg.header)

    def _publish_debug_image(self, frame, result, header) -> None:
        import cv2
        annotated = frame.copy()
        if result.pixel is not None:
            u, v = result.pixel
            radius = int(result.radius_px or 5)
            cv2.circle(annotated, (int(u), int(v)), radius, (0, 255, 0), 2)
        status = "OK" if result.valid else "LOST"
        cv2.putText(annotated, f"ball: {status}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        debug_msg.header = header
        self.pub_debug.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BallTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
