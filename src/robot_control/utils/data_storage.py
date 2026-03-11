"""Functions for storing robot observation and action data to JSON files."""

import json
from pathlib import Path

import numpy as np
import torch

from robot_control.utils.kinematics_utils import trans_mat_to_pos_quat, gripper_raw_to_qpos


def store_state_data(
    obs: torch.Tensor,
    curr_qpos: np.ndarray,
    action_fingertip: torch.Tensor,
    action_qpos: np.ndarray,
    timestamp: float,
    robot_obs_record_dir: Path,
    robot_action_record_dir: Path,
) -> None:
    """Store residual-mode observation and action data.

    Args:
        obs: Observation tensor of shape (1, 35).
        curr_qpos: Current joint positions.
        action_fingertip: Fingertip action tensor of shape (1, 8).
        action_qpos: Joint-space action array of shape (8,).
        timestamp: Capture timestamp for filename.
        robot_obs_record_dir: Directory for observation JSON files.
        robot_action_record_dir: Directory for action JSON files.
    """
    stored_obs = {
        "obs.fingertip_pos": obs[0, :3].cpu().numpy().tolist(),
        "obs.fingertip_quat": obs[0, 3:7].cpu().numpy().tolist(),
        "obs.gripper": obs[0, 7].cpu().numpy().tolist(),
        "obs.fingertip_pos_rel_fixed": obs[0, 8:11].cpu().numpy().tolist(),
        "obs.fingertip_pos_rel_held": obs[0, 11:14].cpu().numpy().tolist(),
        "obs.ee_linvel_fd": obs[0, 14:17].cpu().numpy().tolist(),
        "obs.ee_angvel_fd": obs[0, 17:20].cpu().numpy().tolist(),
        "obs.qpos": curr_qpos.tolist(),
    }

    with open(robot_obs_record_dir / f"{timestamp:.3f}.json", "w") as f:
        json.dump(stored_obs, f, indent=4)

    stored_action = {
        "base_action.fingertip_pos": obs[0, 20:23].cpu().numpy().tolist(),
        "base_action.fingertip_quat": obs[0, 23:27].cpu().numpy().tolist(),
        "base_action.gripper": obs[0, 27].cpu().numpy().tolist(),
        "action.fingertip_pos": action_fingertip[0, :3].cpu().numpy().tolist(),
        "action.fingertip_quat": action_fingertip[0, 3:7].cpu().numpy().tolist(),
        "action.gripper": action_fingertip[0, 7].cpu().numpy().tolist(),
        "action.qpos": action_qpos[:7].tolist(),
    }

    with open(robot_action_record_dir / f"{timestamp:.3f}.json", "w") as f:
        json.dump(stored_action, f, indent=4)


def store_robot_data(
    trans_out: dict,
    qpos_out: dict,
    gripper_out: dict,
    action_qpos_out: dict,
    action_trans_out: dict,
    action_gripper_out: dict,
    robot_obs_record_dir: Path,
    robot_action_record_dir: Path,
    force_out: dict = None,
    bimanual: bool = False,
    gripper_enable: bool = False,
) -> None:
    """Store teleop-mode robot observation and action data.

    Args:
        trans_out: End-effector transform dictionary.
        qpos_out: Joint positions dictionary.
        gripper_out: Gripper state dictionary.
        action_qpos_out: Commanded joint positions dictionary.
        action_trans_out: Commanded end-effector transform dictionary.
        action_gripper_out: Commanded gripper state dictionary.
        robot_obs_record_dir: Directory to save observation JSON files.
        robot_action_record_dir: Directory to save action JSON files.
        force_out: Optional force/torque data dictionary.
        bimanual: Whether using dual-arm setup.
        gripper_enable: Whether gripper is enabled.
    """
    res_obs = {}
    if bimanual:
        qpos_L = qpos_out["left_value"]
        trans_L = trans_out["left_value"]
        qpos_R = qpos_out["right_value"]
        trans_R = trans_out["right_value"]

        pos_l, rot_l = trans_mat_to_pos_quat(trans_L)
        res_obs["obs.qpos.left"] = qpos_L.tolist()
        res_obs["obs.ee_pos.left"] = pos_l.tolist()
        res_obs["obs.ee_quat.left"] = rot_l.tolist()

        pos_r, rot_r = trans_mat_to_pos_quat(trans_R)
        res_obs["obs.qpos.right"] = qpos_R.tolist()
        res_obs["obs.ee_pos.right"] = pos_r.tolist()
        res_obs["obs.ee_quat.right"] = rot_r.tolist()

        if gripper_enable:
            res_obs["obs.gripper_qpos.left"] = gripper_out["left_value"].tolist()
            res_obs["obs.gripper_qpos.right"] = gripper_out["right_value"].tolist()
    else:
        ee2base = trans_out["value"]
        pos, quat = trans_mat_to_pos_quat(ee2base)
        res_obs["obs.ee_pos"] = pos.tolist()
        res_obs["obs.ee_quat"] = quat.tolist()

        qpos = qpos_out["value"]
        res_obs["obs.qpos"] = qpos.tolist()

        if gripper_enable:
            gripper = gripper_out["value"][0]
            res_obs["obs.gripper_qpos"] = gripper_raw_to_qpos(gripper)

        if force_out is not None:
            res_obs["obs.force"] = force_out["value"].tolist()

    with open(robot_obs_record_dir / f"{trans_out['capture_time']:.3f}.json", "w") as f:
        json.dump(res_obs, f, indent=4)

    # save action in a different file
    res_action = {}
    if bimanual:
        action_qpos_L = action_qpos_out["left_value"]
        action_trans_L = action_trans_out["left_value"]
        action_qpos_R = action_qpos_out["right_value"]
        action_trans_R = action_trans_out["right_value"]

        pos_l, rot_l = trans_mat_to_pos_quat(action_trans_L)
        res_action["action.qpos.left"] = action_qpos_L.tolist()
        res_action["action.ee_pos.left"] = pos_l.tolist()
        res_action["action.ee_quat.left"] = rot_l.tolist()

        pos_r, rot_r = trans_mat_to_pos_quat(action_trans_R)
        res_action["action.qpos.right"] = action_qpos_R.tolist()
        res_action["action.ee_pos.right"] = pos_r.tolist()
        res_action["action.ee_quat.right"] = rot_r.tolist()

        if gripper_enable:
            res_action["action.gripper_qpos.left"] = action_gripper_out["left_value"].tolist()
            res_action["action.gripper_qpos.right"] = action_gripper_out["right_value"].tolist()
    else:
        action_qpos = action_qpos_out["value"]
        action_trans = action_trans_out["value"]
        res_action["action.qpos"] = action_qpos.tolist()

        pos, rot = trans_mat_to_pos_quat(action_trans)
        res_action["action.ee_pos"] = pos.tolist()
        res_action["action.ee_quat"] = rot.tolist()

        if gripper_enable:
            res_action["action.gripper_qpos"] = action_gripper_out["value"][0]

    with open(robot_action_record_dir / f"{action_qpos_out['capture_time']:.3f}.json", "w") as f:
        json.dump(res_action, f, indent=4)
