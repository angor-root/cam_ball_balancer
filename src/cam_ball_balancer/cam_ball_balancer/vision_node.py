"""ROS2 node: camara + pose del plato + pelota en UN solo proceso.

La imagen NUNCA sale del proceso como Image cruda: se captura, se
procesa y hacia ROS solo salen datos. Esto reemplaza la cadena
publish_camera.py -> /camera/image_raw -> ball_tracker_node +
platform_pose_node, que por WiFi saturaba el enlace (~220 Mbps) y
obligaba a cada nodo a deserializar el cuadro.

Publica (nombres relativos, se namespacean desde el launch):
  ball_position    (cam_ball_balancer_msgs/BallPosition)  x, y en metros, frame del plato
  platform_pose    (cam_ball_balancer_msgs/PlatformPose)  theta_x, theta_y
  vision/stats     (cam_ball_balancer_msgs/VisionStats)   Hz, jitter, latencia, tiempos
  debug_image/compressed (sensor_msgs/CompressedImage)    SOLO si debug_image_hz > 0
  TF camera_link -> plate_link -> ball_link

Todos los mensajes de un mismo cuadro llevan el MISMO stamp (instante
de captura), asi el consumidor puede emparejar pelota y pose.
"""
from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CompressedImage
from tf2_ros import TransformBroadcaster

from cam_ball_balancer_msgs.msg import BallPosition, PlatformPose as PlatformPoseMsg, VisionStats

from .ball_tracker import BallDetectorConfig
from .camera import Camera, CameraConfig
from .platform_pose import PlatformPoseConfig, CameraIntrinsics
from .platform_pose_node import _default_intrinsics_path, _quat_from_rotation_matrix
from .vision_pipeline import PipelineConfig, VisionPipeline, RateStats, draw_debug


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        p = self.declare_parameter

        # Camara
        p("device", "0")
        p("width", 640)
        p("height", 480)
        p("fps", 30.0)
        p("fourcc", "MJPG")
        p("exposure", -1.0)
        p("wb_temperature", -1.0)
        p("camera_intrinsics_path", _default_intrinsics_path())
        # Frames
        p("camera_frame_id", "camera_link")
        p("plate_frame_id", "plate_link")
        p("ball_frame_id", "ball_link")
        p("publish_tf", True)
        # Plato / tags
        p("marker_length_m", 0.03)
        p("corner_marker_ids", [0, 1, 2, 3])
        p("corner_positions_m", [-0.09, -0.09, 0.09, -0.09, 0.09, 0.09, -0.09, 0.09])
        p("aruco_every_n", 1)
        p("pose_hold_max_frames", 10)
        # Pelota
        p("hsv_lower", [5, 120, 120])
        p("hsv_upper", [18, 255, 255])
        p("min_radius_px", 4.0)
        p("max_radius_px", 200.0)
        p("ball_radius_m", 0.02)
        p("use_roi", True)
        p("roi_margin_px", 30)
        p("max_consecutive_misses", 15)
        p("gate_max_speed", 2.0)
        p("gate_min_jump", 0.02)
        # Diagnostico
        p("debug_image_hz", 0.0)
        p("stats_hz", 2.0)

        g = lambda name: self.get_parameter(name).value  # noqa: E731

        intrinsics_path = g("camera_intrinsics_path")
        if intrinsics_path and os.path.isfile(intrinsics_path):
            intrinsics = CameraIntrinsics.load(intrinsics_path)
            self.get_logger().info(f"Intrinsecos: {intrinsics_path}")
        else:
            intrinsics = CameraIntrinsics.identity_guess(g("width"), g("height"))
            self.get_logger().warn(
                f"Sin intrinsecos en {intrinsics_path or '(sin ruta)'}: usando identity_guess(), "
                "angulos y metros seran aproximados."
            )

        flat = g("corner_positions_m")
        pipeline_cfg = PipelineConfig(
            detector=BallDetectorConfig(
                hsv_lower=tuple(g("hsv_lower")),
                hsv_upper=tuple(g("hsv_upper")),
                min_radius_px=g("min_radius_px"),
                max_radius_px=g("max_radius_px"),
            ),
            pose=PlatformPoseConfig(
                marker_length_m=g("marker_length_m"),
                corner_marker_ids=tuple(g("corner_marker_ids")),
                corner_positions_m=tuple((flat[i], flat[i + 1]) for i in range(0, len(flat), 2)),
            ),
            aruco_every_n=g("aruco_every_n"),
            pose_hold_max_frames=g("pose_hold_max_frames"),
            ball_radius_m=g("ball_radius_m"),
            use_roi=g("use_roi"),
            roi_margin_px=g("roi_margin_px"),
            max_consecutive_misses=g("max_consecutive_misses"),
            gate_max_speed=g("gate_max_speed"),
            gate_min_jump=g("gate_min_jump"),
        )
        self.pipeline = VisionPipeline(pipeline_cfg, intrinsics)

        self.camera_frame = g("camera_frame_id")
        self.plate_frame = g("plate_frame_id")
        self.ball_frame = g("ball_frame_id")
        self.publish_tf = g("publish_tf")

        self.pub_ball = self.create_publisher(BallPosition, "ball_position", 10)
        self.pub_pose = self.create_publisher(PlatformPoseMsg, "platform_pose", 10)
        self.pub_stats = self.create_publisher(VisionStats, "vision/stats", 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.debug_period = 1.0 / g("debug_image_hz") if g("debug_image_hz") > 0 else None
        self.pub_debug = (
            self.create_publisher(CompressedImage, "debug_image/compressed", 1)
            if self.debug_period else None
        )
        self.stats_period = 1.0 / max(g("stats_hz"), 0.1)

        self.camera = Camera(CameraConfig(
            device=g("device"), width=g("width"), height=g("height"), fps=g("fps"),
            fourcc=g("fourcc"), exposure=g("exposure"), wb_temperature=g("wb_temperature"),
        ))
        self.get_logger().info(f"Camara (lo que acepto el driver): {self.camera.actual_settings()}")

        self.proc_rate = RateStats()
        self.dropped = 0
        self._running = True
        self.camera.start()
        self._worker = threading.Thread(target=self._loop, name="vision_loop", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        last_seq = -1
        last_stats = last_debug = 0.0
        while self._running and rclpy.ok():
            frame = self.camera.wait_next(last_seq, timeout=1.0)
            if frame is None:
                self.get_logger().warn("Sin cuadros de la camara en 1 s", throttle_duration_sec=5.0)
                continue
            if last_seq >= 0 and frame.seq > last_seq + 1:
                self.dropped += frame.seq - last_seq - 1
            last_seq = frame.seq

            result = self.pipeline.process(frame.image, frame.t_mono)

            t_pub = time.perf_counter()
            stamp = Time(nanoseconds=int(frame.t_wall * 1e9)).to_msg()
            self._publish_data(result, stamp)
            publish_ms = (time.perf_counter() - t_pub) * 1e3

            now = time.monotonic()
            self.proc_rate.tick(now)
            if now - last_stats >= self.stats_period:
                last_stats = now
                self._publish_stats(stamp, result, publish_ms, latency_ms=(now - frame.t_mono) * 1e3)
            if self.debug_period and now - last_debug >= self.debug_period:
                last_debug = now
                self._publish_debug(frame.image, result, stamp)

    def _publish_data(self, result, stamp) -> None:
        pose = result.pose
        pm = PlatformPoseMsg()
        pm.header.stamp = stamp
        pm.header.frame_id = self.camera_frame
        pm.theta_x, pm.theta_y = float(pose.theta_x), float(pose.theta_y)
        pm.tx, pm.ty, pm.tz = float(pose.tx), float(pose.ty), float(pose.tz)
        pm.valid = bool(pose.valid)
        self.pub_pose.publish(pm)

        b = result.ball
        bm = BallPosition()
        bm.header.stamp = stamp
        bm.header.frame_id = self.plate_frame
        bm.x, bm.y, bm.valid = float(b.x), float(b.y), bool(b.valid)
        self.pub_ball.publish(bm)

        if not self.tf_broadcaster:
            return
        transforms = []
        if pose.valid and pose.rvec is not None:
            R, _ = cv2.Rodrigues(pose.rvec)
            qw, qx, qy, qz = _quat_from_rotation_matrix(R)
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.camera_frame
            tf.child_frame_id = self.plate_frame
            tf.transform.translation.x = float(pose.tx)
            tf.transform.translation.y = float(pose.ty)
            tf.transform.translation.z = float(pose.tz)
            tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
            tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
            transforms.append(tf)
        if b.valid:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.plate_frame
            tf.child_frame_id = self.ball_frame
            tf.transform.translation.x = float(b.x)
            tf.transform.translation.y = float(b.y)
            tf.transform.translation.z = -float(self.pipeline.config.ball_radius_m)
            tf.transform.rotation.w = 1.0
            transforms.append(tf)
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

    def _publish_stats(self, stamp, result, publish_ms: float, latency_ms: float) -> None:
        s = self.proc_rate.summary()
        m = VisionStats()
        m.header.stamp = stamp
        m.header.frame_id = self.camera_frame
        m.rate_hz = s["hz"]
        m.period_ms = s["period_ms"]
        m.jitter_ms = s["jitter_ms"]
        m.period_max_ms = s["period_max_ms"]
        m.camera_hz = self.camera.rate.summary()["hz"]
        m.aruco_ms = result.timings_ms.get("aruco_ms", 0.0)
        m.ball_ms = result.timings_ms.get("ball_ms", 0.0)
        m.publish_ms = publish_ms
        m.latency_ms = latency_ms
        m.dropped_frames = self.dropped
        self.pub_stats.publish(m)

    def _publish_debug(self, image, result, stamp) -> None:
        annotated = draw_debug(image, result, self.proc_rate.summary())
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame
        msg.format = "jpeg"
        msg.data = buf.tobytes()
        self.pub_debug.publish(msg)

    def destroy_node(self):
        self._running = False
        self._worker.join(timeout=2.0)
        self.camera.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
