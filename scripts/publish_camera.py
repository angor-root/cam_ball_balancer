#!/usr/bin/env python3
"""Publica frames de una camara real (USB via OpenCV VideoCapture) como
topic ROS2, mismo patron que scripts/publish_synthetic_video.py pero con
hardware real en vez de datos sinteticos.

Si la camara es CSI (Raspberry Pi Camera Module por el conector CSI, no
USB), este script NO sirve tal cual -- cv2.VideoCapture(index) espera un
/dev/videoN expuesto por V4L2. Para CSI en RPi4/Ubuntu normalmente hay que
usar libcamera/picamera2 (que sí puede exponerse como /dev/videoN via
`libcamera` + v4l2loopback, o escribir un publisher con la API de
picamera2 en vez de este script). Confirmar con Caleb que interfaz es
antes de asumir uno u otro -- ver README.

    python3 scripts/publish_camera.py --camera-index 0
    # en otra terminal:
    ros2 launch dsm_vision dsm_vision.launch.py image_topic:=/camera/image_raw
"""
import argparse
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--topic", default="/camera/image_raw")
    parser.add_argument("--frame-id", default="camera_link")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(
            f"No se pudo abrir la camara index={args.camera_index}. "
            "Si es una camara CSI (Pi Camera Module), este script no aplica -- ver docstring."
        )
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    rclpy.init()
    node = Node("camera_publisher")
    pub = node.create_publisher(Image, args.topic, 10)
    bridge = CvBridge()
    node.get_logger().info(f"Publicando camara index={args.camera_index} en {args.topic} @ ~{args.fps}Hz")

    period_s = 1.0 / args.fps
    try:
        while rclpy.ok():
            ok, frame = cap.read()
            if not ok:
                node.get_logger().warn("Frame perdido, reintentando...")
                time.sleep(period_s)
                continue
            msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = args.frame_id
            pub.publish(msg)
            time.sleep(period_s)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
