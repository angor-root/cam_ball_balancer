"""Brings up both vision nodes (+ optionally RViz) with the shared params.

    ros2 launch cam_ball_balancer cam_ball_balancer.launch.py
    ros2 launch cam_ball_balancer cam_ball_balancer.launch.py rviz:=true
    ros2 launch cam_ball_balancer cam_ball_balancer.launch.py image_topic:=/camera/image_raw

RViz defaults to OFF: this is meant to run headless on the robot's
compute (e.g. a Pi running Ubuntu Server, no display) most of the time.
Pass rviz:=true from a machine that actually has a display (e.g. your
dev laptop, watching the same topics over the network) instead of
making that the default and having it fail to spawn on the robot.

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
    pkg_share = FindPackageShare("cam_ball_balancer")
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
        "rviz", default_value="false",
        description="Launch RViz2 with the bundled config. Off by default "
                     "since this launch file is meant to run headless on "
                     "the robot's compute most of the time.",
    )

    ball_tracker_node = Node(
        package="cam_ball_balancer",
        executable="ball_tracker_node",
        name="ball_tracker_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file"), {
            "image_topic": LaunchConfiguration("image_topic"),
        }],
    )

    platform_pose_node = Node(
        package="cam_ball_balancer",
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
