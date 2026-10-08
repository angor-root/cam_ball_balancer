#!/bin/bash
# Arranque del nodo de vision (lo llama el servicio systemd de usuario
# cam-vision.service). Espera a que haya red y camara antes de lanzar:
# si DDS arranca sin WiFi, solo se anuncia en localhost y la laptop no
# ve los topicos.
CAM=/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH0918_20201224_v008-video-index0
PARAMS=${PARAMS:-/home/robot/params_ec1_tablet.yaml}

for i in $(seq 1 60); do
  ip -4 addr show wlan0 2>/dev/null | grep -q "inet " && break
  echo "esperando red ($i)"; sleep 2
done
for i in $(seq 1 30); do
  [ -e "$CAM" ] && break
  echo "esperando camara ($i)"; sleep 1
done

source /opt/ros/jazzy/setup.bash
source /home/robot/cam_ball_balancer/install/setup.bash
exec ros2 launch cam_ball_balancer_cpp vision.launch.py \
  device:="$CAM" debug_image_hz:=5.0 params_file:="$PARAMS"
