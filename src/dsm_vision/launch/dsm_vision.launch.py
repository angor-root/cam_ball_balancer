"""Brings up both vision nodes (+ optionally RViz) with the shared params.

    ros2 launch dsm_vision dsm_vision.launch.py
    ros2 launch dsm_vision dsm_vision.launch.py rviz:=false
    ros2 launch dsm_vision dsm_vision.launch.py image_topic:=/camera/image_raw

Both nodes are launched under their own default node name (ball_tracker_node,
platform_pose_node) so their topics match config/params.yaml as-is; pass a
namespace if this needs to live alongside the rest of the robot's stack.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("dsm_vision")
    default_params = PathJoinSubstitution([pkg_share, "config", "params.yaml"])
    default_rviz = PathJoinSubstitution([pkg_share, "config", "rviz.rviz"])

    image_topic_arg = DeclareLaunchArgument(
        "image_topic", default_value="image_raw",
        description="Camera image topic both nodes subscribe to.",
    )
    params_file_arg = DeclareLaunchArgument(
        "params_file", default_value=default_params,
        description="YAML params file for both nodes.",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="true",
        description="Launch RViz2 with the bundled config.",
    )

    ball_tracker_node = Node(
        package="dsm_vision",
        executable="ball_tracker_node",
        name="ball_tracker_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), {
            "image_topic": LaunchConfiguration("image_topic"),
        }],
    )

    platform_pose_node = Node(
        package="dsm_vision",
        executable="platform_pose_node",
        name="platform_pose_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), {
            "image_topic": LaunchConfiguration("image_topic"),
        }],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", default_rviz],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription([
        image_topic_arg,
        params_file_arg,
        rviz_arg,
        ball_tracker_node,
        platform_pose_node,
        rviz_node,
    ])
