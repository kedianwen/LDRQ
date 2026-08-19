"""Launch the R1 policy node.

    ros2 launch r1_policy_runner policy_node.launch.py \
        engine:=/abs/path/policy.plan mode:=selftest

`params` defaults to the interface descriptor emitted by tools/dump_interface.py
and installed into this package's share directory. It carries the observation
term layout, history depth, action scale and default joint pose -- i.e. every
number the C++ side needs that is a property of the trained policy rather than
of the robot.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "r1_policy_runner"


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory(PKG), "config", "policy_interface.yaml"
    )

    args = [
        DeclareLaunchArgument(
            "engine", description="absolute path to the serialised TensorRT engine (.plan)"
        ),
        DeclareLaunchArgument(
            "params",
            default_value=default_params,
            description="policy interface descriptor (see tools/dump_interface.py)",
        ),
        DeclareLaunchArgument(
            "mode",
            default_value="subscribe",
            description="'subscribe' reads ~/obs; 'selftest' self-drives at control_rate_hz",
        ),
        DeclareLaunchArgument("rate", default_value="50.0", description="control rate in Hz"),
    ]

    node = Node(
        package=PKG,
        executable="r1_policy_node",
        name="r1_policy_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            LaunchConfiguration("params"),
            {
                "engine_path": LaunchConfiguration("engine"),
                "mode": LaunchConfiguration("mode"),
                "control_rate_hz": LaunchConfiguration("rate"),
            },
        ],
    )

    return LaunchDescription(args + [node])
