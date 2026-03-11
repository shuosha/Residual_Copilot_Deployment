"""Backward-compatible re-exports from the split environment modules.

The monolithic RobotEnv has been split into:
- XarmEnv: Base class with shared infrastructure
- TeleopEnv: Gello/keyboard teleop
- PolicyEnv: Autonomous policy rollout
- ReplayEnv: Trajectory replay
- SharedAutonomyEnv: Shared autonomy (online + offline)
"""

from robot_control.env.xarm_env import XarmEnv
from robot_control.env.teleop_env import TeleopEnv
from robot_control.env.policy_env import PolicyEnv
from robot_control.env.replay_env import ReplayEnv
from robot_control.env.shared_autonomy_env import SharedAutonomyEnv

# Legacy aliases
RobotEnv = XarmEnv
ResidualEnv = SharedAutonomyEnv

__all__ = ["XarmEnv", "TeleopEnv", "PolicyEnv", "ReplayEnv", "SharedAutonomyEnv", "ResidualEnv", "RobotEnv"]
