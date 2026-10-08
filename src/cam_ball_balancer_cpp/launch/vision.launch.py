"""vision_node en C++ (camara + pose del plato + pelota, solo datos/TF hacia ROS).

    ros2 launch cam_ball_balancer_cpp vision.launch.py
    ros2 launch cam_ball_balancer_cpp vision.launch.py device:=/dev/video2 debug_image_hz:=5.0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_params = PathJoinSubstitution([FindPackageShare("cam_ball_balancer_cpp"), "config", "params.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("device", default_value="0"),
        DeclareLaunchArgument("debug_image_hz", default_value="0.0"),
        Node(
            package="cam_ball_balancer_cpp",
            executable="vision_node",
            name="vision_node",
            output="screen",
            parameters=[LaunchConfiguration("params_file"), {
                "device": ParameterValue(LaunchConfiguration("device"), value_type=str),
                "debug_image_hz": ParameterValue(LaunchConfiguration("debug_image_hz"), value_type=float),
            }],
        ),
    ])
