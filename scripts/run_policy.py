"""Run autonomous policy rollout.

Example usage:
    python scripts/run_policy.py exp_name --ckpt /path/to/checkpoint
"""

import argparse
import multiprocess as mp
import sys

from robot_control.utils.utils import get_root, kill_stale_multiprocess_helpers
root = get_root(__file__)
sys.path.append(str(root / "logs"))

import numpy as np
from robot_control.env.policy_env import PolicyEnv


if __name__ == '__main__':
    kill_stale_multiprocess_helpers()
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, default='')
    parser.add_argument('--bimanual', action='store_true')
    parser.add_argument('--init_pose', type=list, default=[0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0, 0.0])
    parser.add_argument('--robot_ip', type=str, default="192.168.1.196")
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()

    assert args.name != '', "Please provide a name for the experiment"

    init_poses = [np.array(
        [0.025383975356817245,
        -0.397950679063797,
        0.039473529905080795,
        0.6066429615020752,
        0.1294342428445816,
        1.087607979774475,
        -0.08492714166641235,
        0.0], dtype=np.float32) * 180 / np.pi]

    env = PolicyEnv(
        dp_ckpt_path=args.ckpt,

        exp_name=args.name,
        data_dir="rollout",
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
        init_poses=init_poses,
    )

    env.start()
    env.join()
