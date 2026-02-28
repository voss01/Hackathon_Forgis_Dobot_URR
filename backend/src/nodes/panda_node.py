"""ROS 2 node wrapper for Franka Emika Panda robot arm.

"""

import math
import time
from typing import Optional, List

from rclpy.node import Node
from sensor_msgs.msg import JointState

# TODO: Import Franka-specific message / service / action types once
# the franka_msgs package is available in the workspace.
# Examples:
#   from franka_msgs.action import Move, Grasp
#   from franka_msgs.srv import SetLoad, SetFullCollisionBehavior
#   from control_msgs.action import FollowJointTrajectory

# Panda has 7 DOF joints
PANDA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]


class PandaNode(Node):
    """
    Thin ROS 2 node for Franka Emika Panda.

    Follows the same pattern as RobotNode (UR) and DobotNova5Node:
    - Subscribes to joint state feedback
    - Exposes synchronous motion helpers
    - State queries used by the executor for polling
    """

    def __init__(self):
        super().__init__("panda_node")


    def send_movej(self, target_rad: List[float], accel: float = 1.4, vel: float = 1.05) -> None:
        """Send a joint-space move command.

        TODO: Implement via FollowJointTrajectory action or MoveIt 2.
        """
        raise NotImplementedError("PandaNode.send_movej not yet implemented")

    def send_movel(self, pose: List[float], accel: float = 1.2, vel: float = 0.25) -> None:
        """Send a Cartesian linear move command.

        Args:
            pose: Target pose [x, y, z, rx, ry, rz] in meters and radians.

        TODO: Implement via MoveIt 2 Cartesian planning or Franka Cartesian controller.
        """
        raise NotImplementedError("PandaNode.send_movel not yet implemented")

    # ── State queries ────────────────────────────────────────────────────

    def get_joint_positions(self) -> Optional[List[float]]:
        """Return current joint positions in radians (7 DOF), or None."""
        if self._joint_positions is None:
            return None
        return [self._joint_positions.get(name, 0.0) for name in PANDA_JOINT_NAMES]

    def get_joint_positions_deg(self) -> Optional[List[float]]:
        """Return current joint positions in degrees, or None."""
        rads = self.get_joint_positions()
        if rads is None:
            return None
        return [math.degrees(r) for r in rads]

    def joints_at_target(self, target_rad: List[float], tolerance: float = 0.02) -> bool:
        """Check if current joints are within tolerance of target (radians)."""
        current = self.get_joint_positions()
        if current is None:
            return False
        return all(abs(c - t) < tolerance for c, t in zip(current, target_rad))

    def get_state_summary(self) -> dict:
        return {
            "timestamp": time.time(),
            "connected": self._joint_positions is not None,
            "joints_deg": self.get_joint_positions_deg(),
            "io": {"digital_in": [], "digital_out": []},
        }
