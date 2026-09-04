"""Bring up bridge + policy wired together.

USE THIS, not the two individual launch files, whenever both nodes run.

Each node names its topics under its own namespace (`~/obs` is
`/r1_hw_bridge/obs` for the bridge but `/r1_policy_node/obs` for the policy), so
without the remappings below the two nodes come up, log happily, and never
exchange a single message. The failure is silent in both directions:

  * the policy node waits on ~/obs forever and reports "filling observation
    history" once a second, which looks like a warm-up rather than a fault;
  * the bridge's ~/resume path publishes ~/policy_reset into the void, so an
    outage recovery would resume with a history straddling the gap -- exactly
    the out-of-distribution restart the design exists to prevent.

Output is OFF by default. Turning it on is a separate, deliberate act.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

BRIDGE = "r1_hw_bridge"
POLICY = "r1_policy_runner"


def generate_launch_description():
    policy_params = os.path.join(
        get_package_share_directory(POLICY), "config", "policy_interface.yaml")
    bridge_params = os.path.join(
        get_package_share_directory(BRIDGE), "config", "bridge.yaml")

    args = [
        DeclareLaunchArgument("engine", description="absolute path to the .plan"),
        DeclareLaunchArgument("iface", default_value="eth10"),
        DeclareLaunchArgument("enable_output", default_value="false"),
        DeclareLaunchArgument("kp_scale", default_value="1.0"),
        DeclareLaunchArgument("kd_scale", default_value="1.0"),
    ]

    bridge = Node(
        package=BRIDGE, executable="r1_hw_bridge_node", name="r1_hw_bridge",
        output="screen", emulate_tty=True,
        parameters=[bridge_params, {
            "iface": LaunchConfiguration("iface"),
            "enable_output": LaunchConfiguration("enable_output"),
            "kp_scale": LaunchConfiguration("kp_scale"),
            "kd_scale": LaunchConfiguration("kd_scale"),
        }],
        remappings=[
            ("joint_target", "/r1_policy_node/joint_target"),
            ("action", "/r1_policy_node/action"),
        ],
    )

    policy = Node(
        package=POLICY, executable="r1_policy_node", name="r1_policy_node",
        output="screen", emulate_tty=True,
        parameters=[policy_params, {
            "engine_path": LaunchConfiguration("engine"),
            "mode": "subscribe",
        }],
        remappings=[
            ("~/obs", "/r1_hw_bridge/obs"),
            ("~/policy_reset", "/r1_hw_bridge/policy_reset"),
        ],
    )

    return LaunchDescription(args + [bridge, policy])
