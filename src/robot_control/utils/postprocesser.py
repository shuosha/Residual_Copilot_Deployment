from __future__ import annotations

from typing import Union, Optional, Dict, List
from pathlib import Path
import argparse
import os
import subprocess
import numpy as np
import glob
import cv2
import torch
import shutil
from PIL import Image
import json
import sys
import kornia

from robot_control.utils.utils import get_root, mkdir
root = get_root(__file__)

from robot_control.utils.kinematics_utils import eef_to_fingertip

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*torch.cross without specifying the dim arg is deprecated.*"
)

def _load_sorted_timestamps_json(dir_path: Path) -> List[float]:
    """Return sorted list of float timestamps from *.json filenames."""
    files = sorted(dir_path.glob("*.json"))
    return [float(p.stem) for p in files]


def _nearest_bracketing_indices(ts: np.ndarray, t: float) -> Optional[tuple[int, int, float]]:
    """
    Given sorted 1D timestamps ts and a query time t,
    return (i0, i1, w) such that ts[i0] <= t <= ts[i1], and
    w is interpolation weight in [0,1]. If t is outside range, return None.
    """
    if ts.size < 2:
        return None
    j = np.searchsorted(ts, t, side="left")
    if j == 0:
        return 0, 1, 0.0 #  use first two samples, weight 0
    if j >= ts.size:
        return ts.size-2, ts.size-1, 1.0 # clamp to last two samples
    i0, i1 = j - 1, j
    denom = (ts[i1] - ts[i0]) if ts[i1] > ts[i0] else 1e-10
    w = (t - ts[i0]) / denom
    return i0, i1, float(w)


def _slerp_quat_list(q1, q2, w: float) -> list:
    """q1, q2: list/array flattened, multiples of 4. Return flattened list."""
    a = np.array(q1, dtype=np.float32).reshape(-1, 4)
    b = np.array(q2, dtype=np.float32).reshape(-1, 4)
    qa = kornia.geometry.quaternion.Quaternion(torch.tensor(a))
    qb = kornia.geometry.quaternion.Quaternion(torch.tensor(b))
    out = qa.slerp(qb, w).data.detach().cpu().numpy().reshape(-1).tolist()
    return out


def _is_quat_key(key_main: str) -> bool:
    return key_main in {"fingertip_quat", "ee_quat.left", "ee_quat.right"}


def _split_prefix(k: str):
    if k.startswith("action."):
        return "action.", k[len("action."):]
    if k.startswith("base_action."):
        return "base_action.", k[len("base_action."):]
    raise AssertionError(f"Unexpected key prefix: {k}")


def synchronize_timesteps(
    data_path: str,
    recording_dirs: Optional[Dict[str, List[str]]] = None,
    bimanual: bool = False,
):
    """
    Build synchronized episodes by matching camera frames to the closest
    robot obs/action times and interpolating robot values at each master frame time.

    Supports method markers: if a recording contains teleop.txt, residual_copilot.txt, or
    residual_bc.txt, episodes are sorted into per-method output directories.

    data_path: full path to the recording root directory.
    Example: '/home/user/projects/logs/data/20251029_mnet'
    """
    base = Path(data_path).expanduser().resolve()

    # Discover recordings if not provided
    if recording_dirs is None:
        # folders directly under .../<name> that are not reserved
        candidates = [p for p in sorted(base.iterdir()) if p.is_dir()]
        # filter out reserved names robustly using Path.name
        reserved = {"calibration", "robot_obs", "robot_action", "infos"}
        action_names = [p.name for p in candidates if p.name not in reserved]
        recording_dirs = {base.name: action_names}

    name_save = base.name + "_processed"
    main_root = base.parent / name_save

    methods = ("teleop", "residual_copilot", "residual_bc")

    data_dir = main_root / "data"
    mkdir(data_dir, overwrite=True, resume=True)

    episode_idx = {m: 0 for m in methods}
    infos = {m: {} for m in methods}
    for recording_name, action_name_list in recording_dirs.items():
        rec_dir = base
        calibration_dir = rec_dir / "calibration"
        robot_obs_dir = rec_dir / "robot_obs"
        robot_action_dir = rec_dir / "robot_action"

        # Preload robot timestamp arrays (sorted)
        obs_ts = np.array(_load_sorted_timestamps_json(robot_obs_dir), dtype=np.float64)
        act_ts = np.array(_load_sorted_timestamps_json(robot_action_dir), dtype=np.float64)
        if obs_ts.size < 2 or act_ts.size < 2:
            print("[Warning] Skipping recording: insufficient robot streams.")
            continue

        for action_name in action_name_list:
            action_name = Path(action_name).name  # normalize
            action_dir = rec_dir / action_name
            if not action_dir.exists():
                print(f"[Warning] Missing action dir: {action_dir}")
                continue

            # Detect number of cameras
            cam_src_dirs = sorted(
                [p for p in (rec_dir / action_name).iterdir() if p.is_dir() and p.name.startswith("camera_")]
            )
            if not cam_src_dirs:
                raise ValueError(f"No camera dirs found in {rec_dir}/{action_name_list[0]}")
            num_cams = max(1, len(cam_src_dirs))

            # Read per-frame camera timestamps (list of per-cam times per line)
            ts_file = action_dir / "timestamps.txt"
            if not ts_file.exists():
                print(f"[Warning] Missing timestamps: {ts_file}")
                continue
            with open(ts_file, "r") as f:
                lines = f.readlines()
            action_ts = [[float(x) for x in line.split()[-num_cams:]] for line in lines]
            if not action_ts:
                continue

            method = None
            for m in methods:
                if (action_dir / f"{m}.txt").exists():
                    method = m
                    break
            if method is None:
                print(f"[Warning] No method marker found in {action_dir}, skipping.")
                continue
            out_root = main_root / method
            mkdir(out_root, overwrite=False, resume=True)

            # Prepare episode output dirs
            ep_dir = out_root / f"episode_{episode_idx[method]:04d}"
            if (ep_dir / "timestamps.txt").exists():
                print(f"[Skip] {ep_dir} already processed.")
                episode_idx[method] += 1
                continue
            else:
                print(f"[Processing] {ep_dir} with {num_cams} cameras...")

            mkdir(ep_dir, overwrite=True, resume=False)
            if calibration_dir.exists():
                shutil.copytree(calibration_dir, ep_dir / "calibration", dirs_exist_ok=True)

            cam_dirs = []
            for cam in range(num_cams):
                cam_dir = ep_dir / f"camera_{cam}"
                mkdir(cam_dir / "rgb", overwrite=True, resume=False)
                mkdir(cam_dir / "depth", overwrite=True, resume=False)
                cam_dirs.append(cam_dir)

            robot_out_dir = ep_dir / "robot"
            mkdir(robot_out_dir, overwrite=True, resume=False)

            # Match and write frames
            with open(ep_dir / "timestamps.txt", "w") as ts_out:
                for t, per_cam_times in enumerate(action_ts):
                    master_t = per_cam_times[0]
                    # stabilize other cam times by picking the closest within local window t-1..t+1
                    chosen_times = [master_t]
                    for cam in range(1, num_cams):
                        local_idxs = range(max(t - 1, 0), min(t + 2, len(action_ts)))
                        idx_best = min(local_idxs, key=lambda i: abs(action_ts[i][cam] - master_t))
                        chosen_times.append(action_ts[idx_best][cam])

                    # Interp robot obs at master_t
                    br_obs = _nearest_bracketing_indices(obs_ts, master_t)
                    if br_obs is None:
                        print(f"[Warning] Skipping frame {t}: insufficient robot data to interpolate (obs).")
                        continue
                    i0, i1, w_obs = br_obs

                    # Interp robot action at master_t
                    br_act = _nearest_bracketing_indices(act_ts, master_t)
                    if br_act is None:
                        print(f"[Warning] Skipping frame {t}: insufficient robot data to interpolate (action).")
                        continue
                    j0, j1, w_act = br_act

                    # Save selected camera frames
                    for cam in range(num_cams):
                        src_rgb = action_dir / f"camera_{cam}" / "rgb" / f"{t:06d}.jpg"
                        src_d   = action_dir / f"camera_{cam}" / "depth" / f"{t:06d}.png"
                        if src_rgb.exists():
                            shutil.copy2(src_rgb, cam_dirs[cam] / "rgb" / src_rgb.name)
                        if src_d.exists():
                            shutil.copy2(src_d, cam_dirs[cam] / "depth" / src_d.name)

                    # Write chosen timestamps line
                    ts_out.write(" ".join(str(x) for x in chosen_times) + "\n")

                    # ----- Interpolate OBS -----
                    obs1 = json.loads((robot_obs_dir / f"{obs_ts[i0]:.3f}.json").read_text())
                    obs2 = json.loads((robot_obs_dir / f"{obs_ts[i1]:.3f}.json").read_text())
                    robot_json = {}

                    for key, v1 in obs1.items():
                        v2 = obs2[key]
                        key_main = key.replace("obs.", "")
                        if bimanual:
                            allowed = {"xy.left","xy.right","ee_pos.left","ee_pos.right",
                                       "ee_quat.left","ee_quat.right","qpos.left","qpos.right",
                                       "gripper_qpos.left","gripper_qpos.right",
                                       "ee_pos", "ee_quat", "gripper_qpos"}
                        else:
                            allowed = {"fingertip_pos","fingertip_quat","gripper",
                                       "fingertip_pos_rel_fixed", "fingertip_pos_rel_held",
                                        "ee_linvel_fd", "ee_angvel_fd",
                                       "force", "qpos",
                                        "ee_pos", "ee_quat", "gripper_qpos"}

                        assert key_main in allowed, f"Unexpected obs key: {key_main}"

                        if _is_quat_key(key_main):
                            robot_json[key] = _slerp_quat_list(v1, v2, w_obs)
                        else:
                            a = np.asarray(v1, dtype=np.float32)
                            b = np.asarray(v2, dtype=np.float32)
                            robot_json[key] = (a * (1 - w_obs) + b * w_obs).tolist()

                    # ----- Interpolate ACTION -----
                    act1 = json.loads((robot_action_dir / f"{act_ts[j0]:.3f}.json").read_text())
                    act2 = json.loads((robot_action_dir / f"{act_ts[j1]:.3f}.json").read_text())

                    for key, v1 in act1.items():
                        v2 = act2[key]
                        _, key_main = _split_prefix(key)   # accepts action.* and base_action.*

                        if bimanual:
                            allowed = {"xy.left","xy.right","ee_pos.left","ee_pos.right",
                                       "ee_quat.left","ee_quat.right","qpos.left","qpos.right",
                                       "gripper_qpos.left","gripper_qpos.right"}
                        else:
                            allowed = {"fingertip_pos","fingertip_quat","gripper", "qpos",
                                        "ee_pos", "ee_quat", "gripper_qpos"}
                        assert key_main in allowed, f"Unexpected action key: {key_main}"

                        if _is_quat_key(key_main):
                            robot_json[key] = _slerp_quat_list(v1, v2, w_act)
                        else:
                            a = np.asarray(v1, dtype=np.float32)
                            b = np.asarray(v2, dtype=np.float32)
                            robot_json[key] = (a * (1 - w_act) + b * w_act).tolist()

                    (robot_out_dir / f"{t:06d}.json").write_text(json.dumps(robot_json, indent=4))

                if (action_dir / "failed.txt").exists():
                    infos[method][f"episode_{episode_idx[method]:04d}"] = {
                        "status": "failed",
                        "eps_length": t,
                        "clock_time": float(t / 15.0)
                    }
                else:
                    infos[method][f"episode_{episode_idx[method]:04d}"] = {
                        "status": "success",
                        "eps_length": t,
                        "clock_time": float(t / 15.0)
                    }

            episode_idx[method] += 1

    # Only save info for methods that had episodes
    infos_filtered = {m: v for m, v in infos.items() if v}
    (data_dir / "infos.json").write_text(json.dumps(infos_filtered, indent=4))


def pack_episode_trajectories(
    data_path: str,
    method: str = "teleop",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Parse per-episode robot JSON files and pack them into numpy arrays.

    Reads from: {data_path}_processed/{method}/episode_*/robot/*.json
    Saves to:   {data_path}_processed/data/{method}_data.npy

    Returns dict of {episode_name: {key: (T, D) ndarray}}.
    """

    data_dir = Path(data_path).expanduser().resolve()
    if not data_dir.name.endswith("_processed"):
        data_dir = data_dir.parent / f"{data_dir.name}_processed"

    out_filename = f"{method}_data.npy"
    out_path = data_dir / "data" / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_eps_dir = data_dir / method
    if not all_eps_dir.exists():
        print(f"[Warning] No {method}/ directory found in {data_dir}")
        return {}
    episodes = sorted([p for p in all_eps_dir.glob("episode_*") if p.is_dir()])

    def _as1d(x, expect_len: int, key: str):
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size != expect_len:
            raise ValueError(f"{key} expected len {expect_len}, got {arr.size}")
        return arr

    data: Dict[str, Dict[str, np.ndarray]] = {}

    for ep_dir in episodes:
        has_force = False
        robot_dir = ep_dir / "robot"
        if not robot_dir.exists():
            continue
        files = sorted(robot_dir.glob("*.json"), key=lambda p: int(p.stem))
        if not files:
            continue

        ts, obs_pos, obs_quat, obs_grip, obs_fixed, obs_held, obs_linvel, obs_angvel, obs_f, obs_qpos = [], [], [], [], [], [], [], [], [], []
        act_pos, act_quat, act_grip, act_qpos = [], [], [], []
        base_pos, base_quat, base_grip = [], [], []

        for f in files:
            t = float(int(f.stem))
            obj = json.loads(f.read_text())

            o_pos  = _as1d(obj["obs.fingertip_pos"],   3, "obs.fingertip_pos")
            o_quat = _as1d(obj["obs.fingertip_quat"],  4, "obs.fingertip_quat")
            o_grip = _as1d(obj["obs.gripper"], 1, "obs.gripper")
            o_fixed = _as1d(obj["obs.fingertip_pos_rel_fixed"], 3, "obs.fingertip_pos_rel_fixed")
            o_held = _as1d(obj["obs.fingertip_pos_rel_held"], 3, "obs.fingertip_pos_rel_held")
            o_linvel = _as1d(obj["obs.ee_linvel_fd"], 3, "obs.ee_linvel_fd")
            o_angvel = _as1d(obj["obs.ee_angvel_fd"], 3, "obs.ee_angvel_fd")
            o_qpos = _as1d(obj["obs.qpos"],     7, "obs.qpos")

            a_pos  = _as1d(obj["action.fingertip_pos"],   3, "action.fingertip_pos")
            a_quat = _as1d(obj["action.fingertip_quat"],  4, "action.fingertip_quat")
            a_grip = _as1d(obj["action.gripper"], 1, "action.gripper")
            a_qpos = _as1d(obj["action.qpos"],     7, "action.qpos")

            b_pos = _as1d(obj["base_action.fingertip_pos"],   3, "base_action.fingertip_pos")
            b_quat = _as1d(obj["base_action.fingertip_quat"],  4, "base_action.fingertip_quat")
            b_grip = _as1d(obj["base_action.gripper"], 1, "base_action.gripper")

            o_f = _as1d(obj["obs.force"], 6, "obs.force") if "obs.force" in obj else None
            if o_f is not None:
                obs_f.append(o_f)
                has_force = True

            ts.append([t])
            obs_pos.append(o_pos); obs_quat.append(o_quat); obs_grip.append(o_grip); obs_qpos.append(o_qpos)
            obs_fixed.append(o_fixed); obs_held.append(o_held);
            obs_linvel.append(o_linvel); obs_angvel.append(o_angvel)
            act_pos.append(a_pos); act_quat.append(a_quat); act_grip.append(a_grip); act_qpos.append(a_qpos)
            base_pos.append(b_pos); base_quat.append(b_quat); base_grip.append(b_grip)


        ep_key = ep_dir.name
        data[ep_key] = {
            "obs.fingertip_pos"    : np.asarray(obs_pos, dtype=np.float32),
            "obs.fingertip_quat"   : np.asarray(obs_quat, dtype=np.float32),
            "obs.gripper"    : np.asarray(obs_grip, dtype=np.float32),
            "obs.fingertip_pos_rel_fixed": np.asarray(obs_fixed, dtype=np.float32),
            "obs.fingertip_pos_rel_held": np.asarray(obs_held, dtype=np.float32),
            "obs.ee_linvel_fd": np.asarray(obs_linvel, dtype=np.float32),
            "obs.ee_angvel_fd": np.asarray(obs_angvel, dtype=np.float32),
            "obs.qpos"       : np.asarray(obs_qpos, dtype=np.float32),
            "action.fingertip_pos" : np.asarray(act_pos, dtype=np.float32),
            "action.fingertip_quat": np.asarray(act_quat, dtype=np.float32),
            "action.gripper" : np.asarray(act_grip, dtype=np.float32),
            "action.qpos"    : np.asarray(act_qpos, dtype=np.float32),
            "base_action.fingertip_pos" : np.asarray(base_pos, dtype=np.float32),
            "base_action.fingertip_quat": np.asarray(base_quat, dtype=np.float32),
            "base_action.gripper" : np.asarray(base_grip, dtype=np.float32),
        }

        if has_force:
            data[ep_key]["obs.force"] = np.asarray(obs_f, dtype=np.float32)

    # ---- save as one big npz ----
    if data:
        np.save(out_path, data)
        print(f"Saved {method} trajectories ({len(data)} episodes) to {out_path}")

    return data
