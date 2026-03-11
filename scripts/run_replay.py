import os
import argparse
import sys

from robot_control.utils.utils import get_root, kill_stale_multiprocess_helpers
root = get_root(__file__)
sys.path.append(str(root / "logs"))

import numpy as np
from robot_control.env.replay_env import ReplayEnv

"""
Example usage:
    python scripts/run_replay.py replay_test data.npz
"""

def load_npz(file_path: str):
    data = np.load(file_path, allow_pickle=True)
    cart_trajs = []
    init_poses = []

    eps_idx = 0
    while True:
        eps_name = f"episode_{eps_idx:04d}"
        if f'{eps_name}/action.eef_pos' not in data:
            break
        cart_traj = np.concatenate((
            data[f'{eps_name}/action.eef_pos'],
            data[f'{eps_name}/action.eef_quat'],
            data[f'{eps_name}/action.gripper']
        ), axis=1)
        init_pose = np.concatenate((
            data[f'{eps_name}/obs.qpos'][0] * 180.0 / np.pi,
            data[f'{eps_name}/obs.gripper'][0]
        ))

        cart_trajs.append(cart_traj)
        init_poses.append(init_pose)
        eps_idx += 1

    return cart_trajs, init_poses

if __name__ == '__main__':
    kill_stale_multiprocess_helpers()
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, default='')
    parser.add_argument('rpz_path', type=str, default='')
    parser.add_argument('--bimanual', action='store_true')
    parser.add_argument('--robot_ip', type=str, default="192.168.1.196")
    parser.add_argument('--fps', type=float, default=10.0, help='camera recording and agent update fps')
    args = parser.parse_args()

    assert args.name != '', "Please provide a name for the experiment"
    assert os.path.exists(args.rpz_path), f"Replay file {args.rpz_path} does not exist"

    cart_trajs, init_poses = load_npz(args.rpz_path)
    assert len(init_poses[0]) == 8, "xarm7 requires 8 DOF initial pose"

    env = ReplayEnv(
        action_trajs=cart_trajs,

        exp_name=args.name,
        data_dir="replay",
        debug=True,

        resolution=(848, 480),
        capture_fps=30,
        record_fps=30,
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
