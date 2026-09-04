"""Bring up the hardware bridge.

Output is OFF by default. Turning it on is a separate, deliberate act:

    ros2 launch r1_hw_bridge bridge.launch.py enable_output:=true

With output off the node still reads the robot, publishes ~/obs and computes
every command, publishing it to ~/cmd_debug -- so the whole chain including the
joint mapping can be verified against a powered robot that cannot move.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("iface", default_value="eth10"),
        DeclareLaunchArgument("enable_output", default_value="false"),
        DeclareLaunchArgument("kp", default_value="40.0"),
        DeclareLaunchArgument("kd", default_value="1.0"),
        DeclareLaunchArgument("cmd_rate_hz", default_value="500.0"),
    ]
    return LaunchDescription(args + [
        Node(
            package="r1_hw_bridge",
            executable="r1_hw_bridge_node",
            name="r1_hw_bridge",
            output="screen",
            parameters=[{
                "iface": LaunchConfiguration("iface"),
                "enable_output": LaunchConfiguration("enable_output"),
                "kp": LaunchConfiguration("kp"),
                "kd": LaunchConfiguration("kd"),
                "cmd_rate_hz": LaunchConfiguration("cmd_rate_hz"),
            }],
            # The policy side needs the matching half of this wiring
            # (~/obs and ~/policy_reset remapped onto /r1_hw_bridge/...).
            # r1_stack.launch.py does both; prefer it when running the pair.
            remappings=[
                ("joint_target", "/r1_policy_node/joint_target"),
                ("action", "/r1_policy_node/action"),
            ],
        ),
    ])
