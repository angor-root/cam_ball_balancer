"""Core perception package for the MT5001 DSM project (ball-on-plate robot).

Everything in this package is plain Python + OpenCV/numpy and has no
dependency on ROS2 — it is meant to be developed and tested at a desk
with just a webcam, per the team plan (Persona C can work without the
robot). The ROS2 nodes in ``ball_tracker_node.py`` / ``platform_pose_node.py``
are thin wrappers around this core.
"""
