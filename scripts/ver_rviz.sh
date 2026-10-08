#!/bin/bash
# Abre RViz en la laptop para ver lo que publica la Raspberry (vision_node).
# Requisitos: laptop y Pi en la misma red y mismo ROS_DOMAIN_ID (0 por defecto).
source /opt/ros/jazzy/setup.bash
source "$(dirname "$0")/../install/setup.bash"   # mensajes propios (VisionStats, etc.)
ros2 daemon stop >/dev/null 2>&1                 # descubrimiento limpio de la red
exec rviz2 -d "$(dirname "$0")/../src/cam_ball_balancer_cpp/config/vision.rviz"
