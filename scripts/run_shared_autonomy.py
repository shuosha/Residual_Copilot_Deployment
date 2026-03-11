"""Run shared autonomy (residual) control — online (gello + residual) or offline (policy + residual).

Example usage:
    # With abbreviated checkpoint names (auto-downloaded from HuggingFace):
    python scripts/run_shared_autonomy.py exp_name --checkpoint_path GearMesh_Residual_Copilot
    python scripts/run_shared_autonomy.py exp_name --checkpoint_path PegInsert_Residual_Copilot --offline
    python scripts/run_shared_autonomy.py exp_name --checkpoint_path NutThread_Residual_Copilot --res_bc_path NutThread_Residual_BC

    # With local paths:
    python scripts/run_shared_autonomy.py exp_name --checkpoint_path /path/to/ckpt
    python scripts/run_shared_autonomy.py exp_name --checkpoint_path /path/to/ckpt --res_bc_path /path/to/bc
"""

import argparse
import multiprocess as mp
import os
import sys

from robot_control.utils.utils import get_root, kill_stale_multiprocess_helpers
root = get_root(__file__)
sys.path.append(str(root / "logs"))

import numpy as np
from robot_control.env.shared_autonomy_env import SharedAutonomyEnv


_TASK_TO_FP = {
    "gearmesh": "gearmesh",
    "peginsert": "peginsert",
    "nutthread": "nutthread",
}

def infer_fp_dir(checkpoint_path):
    """Infer foundation pose directory from checkpoint path or abbreviated name."""
    # Try abbreviated name first (e.g. "GearMesh_Residual_Copilot")
    task_lower = checkpoint_path.split("_")[0].lower()
    if task_lower in _TASK_TO_FP:
        fp_dir = os.path.join("logs", "foundation_pose_dir", _TASK_TO_FP[task_lower])
        if os.path.isdir(fp_dir):
            return fp_dir
    # Fall back to parsing directory name
    basename = os.path.basename(checkpoint_path)
    for task in _TASK_TO_FP:
        if task in basename.lower():
            fp_dir = os.path.join("logs", "foundation_pose_dir", task)
            if os.path.isdir(fp_dir):
                return fp_dir
    return None


if __name__ == '__main__':
    kill_stale_multiprocess_helpers()
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, default='')
    parser.add_argument('--bimanual', action='store_true')
    parser.add_argument('--init_pose', type=list, default=[0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0, 0.0])
    parser.add_argument('--robot_ip', type=str, default="192.168.1.196")
    parser.add_argument('--fp_dir', default=None, type=str, help='foundation pose directory (auto-inferred from checkpoint if not set)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='path to checkpoint or abbreviated name (e.g. GearMesh_Residual_Copilot)')
    parser.add_argument('--res_bc_path', type=str, default=None,
                        help='path to residual BC checkpoint or abbreviated name (e.g. GearMesh_Residual_BC)')
    parser.add_argument('--offline', action='store_true', help='run in offline mode (no gello)')
    args = parser.parse_args()

    assert args.name != '', "Please provide a name for the experiment"

    # Auto-infer fp_dir from checkpoint name if not provided
    if args.fp_dir is None:
        args.fp_dir = infer_fp_dir(args.checkpoint_path)
        if args.fp_dir:
            print(f"Auto-inferred fp_dir: {args.fp_dir}")

    if args.res_bc_path in (None, "None", "none", ""):
        args.res_bc_path = None

    if args.offline:
        init_poses = [np.array([1.7, -16.1, 2.9, 38.4, 1.8, 53.8, -1.0, 0.0], dtype=np.float32)]
    else:
        init_poses = [np.array([5.1, -19.7, 6.8, 42.2, 2.3, 62, 10.1, 0.0], dtype=np.float32)]
    print("Using initial pose:", init_poses)

    data_dir = "shared_autonomy_offline" if args.offline else "shared_autonomy_online"

    env = SharedAutonomyEnv(
        checkpoint_path=args.checkpoint_path,
        foundation_pose_dir=args.fp_dir,
        offline=args.offline,
        res_bc_path=args.res_bc_path,

        exp_name=args.name,
        data_dir=data_dir,
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
