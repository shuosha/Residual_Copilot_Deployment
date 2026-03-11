import argparse
import multiprocess as mp
import sys

from robot_control.utils.utils import get_root, kill_stale_multiprocess_helpers
root = get_root(__file__)
sys.path.append(str(root / "logs"))

import numpy as np
from robot_control.env.teleop_env import TeleopEnv

"""
Example usage:
    python scripts/run_teleop.py teleop_test
"""

if __name__ == '__main__':
    kill_stale_multiprocess_helpers()
    mp.set_start_method('spawn')  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, default='')
    parser.add_argument('--bimanual', action='store_true')
    parser.add_argument('--input_mode', type=str, default='gello', choices=["gello", "keyboard"])
    parser.add_argument('--init_pose', type=list, default=[0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0, 0.0])
    parser.add_argument('--robot_ip', type=str, default="192.168.1.196")
    args = parser.parse_args()

    assert args.name != '', "Please provide a name for the experiment"

    init_poses = [np.array(args.init_pose, dtype=np.float32)]

    env = TeleopEnv(
        exp_name=args.name,
        data_dir="teleop",
        debug=True,

        resolution=(848, 480),
        capture_fps=30,
        record_fps=15,
        perception_process_func=None,

        # robot
        robot_ip=[args.robot_ip],
        bimanual=args.bimanual,
        gripper_enable=True,

        # control
        control_mode="position_control",
        admittance_control=True,
        ema_factor=1.0,
        action_agent_fps=15.0,
        action_receiver=args.input_mode,
        init_poses=init_poses,
    )

    env.start()
    env.join()
