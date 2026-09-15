#!/usr/bin/env python3
"""Publishes the synthetic ball+plate sequence as a ROS2 image topic, so
the two nodes (and RViz) can be exercised end-to-end with zero hardware.

    ros2 run --prefix 'python3' cam_ball_balancer  # not installed as an entry
    point on purpose (it's a dev tool, not part of the package) — run it
    directly instead:

    python3 scripts/publish_synthetic_video.py

Then, in another terminal:
    ros2 launch cam_ball_balancer cam_ball_balancer.launch.py image_topic:=/synthetic/image_raw
"""
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "cam_ball_balancer"))

from cam_ball_balancer.synthetic import generate_ball_sequence, render_plate_with_markers, SyntheticPlateConfig  # noqa: E402


def main():
    rclpy.init()
    node = Node("synthetic_video_publisher")
    pub = node.create_publisher(Image, "/synthetic/image_raw", 10)
    bridge = CvBridge()

    plate_cfg = SyntheticPlateConfig()
    plate_frame, _ = render_plate_with_markers(plate_cfg)

    seq = generate_ball_sequence(num_frames=300, width=plate_cfg.width, height=plate_cfg.height)
    node.get_logger().info("Publishing synthetic ball+plate frames on /synthetic/image_raw @ ~30Hz")

    i = 0
    try:
        while rclpy.ok():
            _t, ball_frame, _gt = seq[i % len(seq)]
            # Composite: start from the plate (with tags), overlay the ball
            # wherever it isn't the plain background color.
            import numpy as np
            frame = plate_frame.copy()
            ball_mask = np.any(ball_frame != (40, 90, 40), axis=-1)
            frame[ball_mask] = ball_frame[ball_mask]

            msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            pub.publish(msg)

            i += 1
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
