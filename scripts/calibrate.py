from __future__ import annotations

import argparse
import copy
import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

# Project modules
import robot_control.perception.calibration.calibration as calib
from robot_control.perception.calibration.ee_transformation import XArmRobot
from robot_control.perception.calibration.realsense_manager import RealSenseManager


# =====================================================================
# ChArUco board settings — update these to match your printed board.
#   size:         number of (columns, rows) of chessboard squares
#   squareLength: side length of each chessboard square (meters)
#   markerLength: side length of each ArUco marker (meters)
#   dictionary:   ArUco dictionary used to generate the board
# Generate a board: https://calib.io/pages/camera-calibration-pattern-generator
# =====================================================================
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
CALIB_BOARD = cv2.aruco.CharucoBoard(
    size=(4, 5),
    squareLength=0.05,
    markerLength=0.036,
    dictionary=ARUCO_DICT,
)
CHECKER_SIZE_M = 0.05
DETECTOR_PARAMS = cv2.aruco.CharucoParameters()
DETECTOR = cv2.aruco.CharucoDetector(CALIB_BOARD, DETECTOR_PARAMS)

FRAME_WIDTH = 848
FRAME_HEIGHT = 480
FRAME_FPS = 30
COLOR_FRAME_FORMAT = rs.format.bgr8
DEPTH_FRAME_FORMAT = rs.format.z16


# =========================
# Robot poses (wrist sweep)
# =========================
INIT_JOINT_DEG = np.array([0, -45, 0, 30, 0, 75, 0])
PREDEFINED_JOINTS_DEG: List[List[float]] = [
    [0, -45, 0, 30, 0, 75, 0],
    [0, -25.5, 0.1, 42.6, -0.1, 68, 0],
    [3.2, -20.6, -2.7, 46.4, -1.1, 66.6, -20.4],
    [-4, -20.4, 3.7, 46.8, 1.3, 67.7, 26.6],
    [-17.7, -29.9, -12, 40.4, -6.5, 69.7, -8.7],
    [-7.4, -20.6, -17.8, 47.9, -6.5, 66.9, -43.7],
    [15.4, -23.3, 14.2, 45.6, 5.8, 68.5, 4.4],
    [4.9, -4, 19.1, 63.7, 1.9, 68.5, 59.8],
    [0, -12.3, 0.2, 54, -0.2, 66.3, 0],
    [0.1, 14.1, 0.3, 88.3, 0.5, 95.7, -0.1],
    [0.1, 19.1, 0.4, 96.1, 0.8, 109.9, 0],
    [-0.3, -50.9, 0.1, 23, -0.3, 57.5, 0],
    [-0.4, -81.1, -0.1, 7.9, -0.4, 53.8, 0],
    [0.1, -28.3, 0.1, 56.9, 0.2, 85.3, 0],
    [-0.3, -17.3, 0.4, 26.5, -0.8, 44, 0.2],
    [7.8, -21.6, 14.5, 63, 5.6, 84.6, 20.5],
    [5.5, -18.1, 21.5, 67.8, -14.4, 95.6, 25.2],
    [4.7, -3.4, 21, 81.2, -19.7, 95.3, 25.3],
    [9.1, 8.3, 17.5, 96.9, -16.1, 116.5, 18.7],
    [13.6, 11.1, 10.7, 69, -15.6, 86.3, 25],
    [15.3, -16.4, 15.3, 35.8, -25.2, 58.9, 46.3],
    [-10.4, -19.1, -17, 32.5, -6.3, 51.2, -22.9],
    [-10.5, 10.7, -11, 66.8, -2.3, 69.9, -20.9],
    [-10.7, 7.2, -10.6, 73, -2.7, 79.5, -20.9],
    [-14.9, -35.2, -19.2, 28.2, -2.6, 48.3, -31],
]


# ==========
# Utilities
# ==========
def _deg_to_rad(arr_deg: np.ndarray) -> np.ndarray:
    return arr_deg / 180.0 * np.pi


def se3_log(T: np.ndarray) -> np.ndarray:
    R_mat, t = T[:3, :3], T[:3, 3]
    rotvec = R.from_matrix(R_mat).as_rotvec()
    return np.hstack((t, rotvec))


def pose_to_matrix(t: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


def handeye_residual(x: np.ndarray, T_As: List[np.ndarray], T_Bs: List[np.ndarray]) -> float:
    t, rotvec = x[:3], x[3:]
    T_A_to_B = pose_to_matrix(t, rotvec)
    total = 0.0
    for T_Ac, T_Bc in zip(T_As, T_Bs):
        T_Ac_est = T_A_to_B @ T_Bc
        delta = np.linalg.inv(T_Ac) @ T_Ac_est
        total += np.linalg.norm(se3_log(delta))
    return total / max(1, len(T_As))


def rs_intrinsics_to_dict(intr: rs.intrinsics) -> dict:
    # Convert pyrealsense2.intrinsics to plain dict (JSON-serializable)
    return {
        "width": intr.width,
        "height": intr.height,
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "model": str(intr.model),
        "coeffs": [float(c) for c in intr.coeffs],
    }


def _get_images(pattern: Path) -> tuple[List[np.ndarray], List[str]]:
    img_names = sorted(glob.glob(str(pattern)))
    images = [cv2.imread(img_name, cv2.IMREAD_COLOR) for img_name in img_names]
    return images, img_names


def _get_ee_poses(ee_pattern: Path) -> tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    ee_files = sorted(glob.glob(str(ee_pattern)))
    R_list, t_list = [], []
    for name in ee_files:
        with open(name) as f:
            ee = json.load(f)
        # quat in [x, y, z, w]
        R_list.append(R.from_quat(ee["quat"]).as_matrix())
        t_list.append(np.array(ee["translation"], dtype=float).reshape(3, 1))
    return R_list, t_list, ee_files


# ==========
# JSON I/O
# ==========
def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=_to_serializable))


def _pair_key(src: str, dst: str) -> str:
    return f"{src}2{dst}"


# =========================================
# Pairwise wrist↔other optimization (images)
# =========================================
def _estimate_wrist2other_via_charuco(
    images_dir: Path,
    other: str,
    wrist_mtx: np.ndarray,
    other_mtx: np.ndarray,
    show_gui: bool = False,
) -> Optional[np.ndarray]:
    """
    Uses paired images saved as:
      wrist_image@pose_{k}.png
      {other}_image@pose_{k}.png
    Returns 4x4 T_wrist2other if ≥1 valid pair; else None.
    """
    wrist_imgs, wnames = _get_images(images_dir / "wrist_image@pose_*.png")
    other_imgs, onames = _get_images(images_dir / f"{other}_image@pose_*.png")

    if not wrist_imgs or not other_imgs:
        return None

    if len(wrist_imgs) != len(other_imgs):
        print(f"[compose] pair count mismatch (wrist:{len(wrist_imgs)} vs {other}:{len(other_imgs)}) → using min length")
        n = min(len(wrist_imgs), len(other_imgs))
        wrist_imgs, wnames = wrist_imgs[:n], wnames[:n]
        other_imgs, onames = other_imgs[:n], onames[:n]

    T_As: List[np.ndarray] = []
    T_Bs: List[np.ndarray] = []

    for wi, oi, wn, on in zip(wrist_imgs, other_imgs, wnames, onames):
        if wi is None or oi is None:
            print(f"[compose] unreadable images:\n  {wn}\n  {on}")
            continue

        # Wrist Charuco (Board->WristCam)
        gray_w = cv2.cvtColor(wi, cv2.COLOR_BGR2GRAY)
        cc_w, ids_w, mc_w, mid_w = DETECTOR.detectBoard(gray_w)
        if cc_w is None or len(cc_w) < 4:
            print(f"[compose] not enough ChArUco corners in {wn}, skipping")
            continue
        ok_w, rvec_w, tvec_w = cv2.aruco.estimatePoseCharucoBoard(
            cc_w, ids_w, CALIB_BOARD, wrist_mtx, np.zeros(5), rvec=None, tvec=None
        )
        if not ok_w:
            print(f"[compose] pose failure (wrist) in {wn}, skipping")
            continue
        Rw, _ = cv2.Rodrigues(rvec_w)
        Tw = calib.compose_transform(Rw, tvec_w.flatten())

        # Other Charuco (Board->OtherCam)
        gray_o = cv2.cvtColor(oi, cv2.COLOR_BGR2GRAY)
        cc_o, ids_o, mc_o, mid_o = DETECTOR.detectBoard(gray_o)
        if cc_o is None or len(cc_o) < 4:
            print(f"[compose] not enough ChArUco corners in {on}, skipping")
            continue
        ok_o, rvec_o, tvec_o = cv2.aruco.estimatePoseCharucoBoard(
            cc_o, ids_o, CALIB_BOARD, other_mtx, np.zeros(5), rvec=None, tvec=None
        )
        if not ok_o:
            print(f"[compose] pose failure (other) in {on}, skipping")
            continue
        Ro, _ = cv2.Rodrigues(rvec_o)
        To = calib.compose_transform(Ro, tvec_o.flatten())

        T_As.append(Tw)
        T_Bs.append(To)

        if show_gui:
            vis_w = cv2.aruco.drawDetectedMarkers(gray_w.copy(), mc_w, mid_w)
            cv2.drawFrameAxes(vis_w, wrist_mtx, np.zeros(5), rvec_w, tvec_w, CHECKER_SIZE_M * 2)
            vis_o = cv2.aruco.drawDetectedMarkers(gray_o.copy(), mc_o, mid_o)
            cv2.drawFrameAxes(vis_o, other_mtx, np.zeros(5), rvec_o, tvec_o, CHECKER_SIZE_M * 2)
            combo = np.hstack((vis_w, vis_o))
            cv2.imshow(f"Charuco Wrist | {other} (any key to continue)", combo)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if not T_As:
        return None

    # Initial guess and solve
    x0 = se3_log(T_As[0] @ np.linalg.inv(T_Bs[0]))
    res = minimize(lambda x: np.sum(handeye_residual(x, T_As, T_Bs)), x0)
    t_opt, r_opt = res.x[:3], res.x[3:]
    return pose_to_matrix(t_opt, r_opt)


# ================
# Main calibrator
# ================
@dataclass
class HandEyeCalibrator:
    work_dir: Path
    robot_ip: Optional[str]
    show_gui: bool = True
    init_robot: bool = True

    robot: Optional[XArmRobot] = field(init=False, default=None)

    def __post_init__(self):
        self.work_dir = Path(self.work_dir).expanduser()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # image organization
        img_root = self.work_dir / "calibration_images"
        self.handeye_images_dir = img_root / "handeye"     # wrist images + EE poses
        self.compose_images_dir = img_root / "compose"     # paired wrist/other images for compose optimization
        self.marked_images_dir = img_root / "marked"       # visualizations (what you saw)

        self.handeye_images_dir.mkdir(parents=True, exist_ok=True)
        self.compose_images_dir.mkdir(parents=True, exist_ok=True)
        self.marked_images_dir.mkdir(parents=True, exist_ok=True)

        self.extr_path = self.work_dir / "extrinsics.json"
        self.intr_path = self.work_dir / "intrinsics.json"

        if self.init_robot and self.robot_ip is not None:
            self.robot = XArmRobot(ip=self.robot_ip, control_frequency=500.0, max_delta=0.01)

    # 1) fixed calibration: per-camera board2cam + intrinsics (+ save visualization)
    def fixed_calibration(self, serials: Dict[str, str]) -> None:
        extr = _read_json(self.extr_path)
        intr = _read_json(self.intr_path)
        extr.setdefault("board2cam", {})

        for cam_name, serial in serials.items():
            cam = RealSenseManager(
                serial=serial,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
                fps=FRAME_FPS,
                depth_format=DEPTH_FRAME_FORMAT,
                color_format=COLOR_FRAME_FORMAT,
            )
            try:
                cam.poll_frames()
                bgr = cam.get_color_image()
                if bgr is None:
                    raise RuntimeError(f"failed to read frame from camera '{cam_name}'")
                # Save one example image for compose pairing/reference
                cv2.imwrite(str(self.compose_images_dir / f"{cam_name}_image@pose_0.png"), bgr)

                # Intrinsics
                intr_rs = cam.get_color_intrinsic()
                intr_dict = rs_intrinsics_to_dict(intr_rs)
                mtx = np.array([[intr_dict["fx"], 0, intr_dict["ppx"]],
                                [0, intr_dict["fy"], intr_dict["ppy"]],
                                [0, 0, 1]], dtype=float)

                # Detect Charuco → Board->Cam
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                cc, ids, mc, mid = DETECTOR.detectBoard(gray)
                if cc is None or len(cc) < 4:
                    if self.show_gui:
                        cv2.imshow(f"{cam_name}: Charuco not detected", bgr)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    raise RuntimeError(f"Charuco not detected for camera '{cam_name}'.")

                ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    cc, ids, CALIB_BOARD, mtx, np.zeros(5), None, None
                )
                if not ok:
                    raise RuntimeError(f"Charuco pose failed for camera '{cam_name}'.")
                R_cam, _ = cv2.Rodrigues(rvec)
                T_board2cam = calib.compose_transform(R_cam, tvec.flatten())

                # Write results
                extr["board2cam"][cam_name] = np.asarray(T_board2cam, dtype=float).tolist()
                intr[cam_name] = intr_dict

                # Save visualization
                vis = cv2.aruco.drawDetectedMarkers(bgr.copy(), mc, mid)
                cv2.imwrite(str(self.marked_images_dir / f"{cam_name}_fixed_calibration_marker.png"), vis)
                vis = bgr.copy()
                cv2.drawFrameAxes(vis, mtx, np.zeros(5), rvec, tvec, CHECKER_SIZE_M * 2)
                cv2.imwrite(str(self.marked_images_dir / f"{cam_name}_fixed_calibration_axes.png"), vis)
                if self.show_gui:
                    cv2.imshow(f"{cam_name} Charuco", vis)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            finally:
                cam.stop()

        _write_json(self.extr_path, extr)
        _write_json(self.intr_path, intr)
        cv2.destroyAllWindows()

    # 2a) wrist data capture (images + EE poses)
    def capture_wrist_images(self, wrist_serial: Optional[str]) -> None:
        if self.robot is None:
            raise RuntimeError("capture_wrist_images() requires a robot; init with init_robot=True.")

        print("Initializing robot for wrist-image capture...")
        self.robot.move_and_wait_robot(_deg_to_rad(INIT_JOINT_DEG))

        wrist_cam = RealSenseManager(
            serial=wrist_serial,
            width=FRAME_WIDTH, height=FRAME_HEIGHT,
            fps=FRAME_FPS,
            depth_format=DEPTH_FRAME_FORMAT, color_format=COLOR_FRAME_FORMAT,
        )
        try:
            wrist_cam.poll_frames()
            for idx, joint_deg in enumerate(PREDEFINED_JOINTS_DEG):
                print(f"Capturing {idx+1}/{len(PREDEFINED_JOINTS_DEG)}...")
                self.robot.move_and_wait_robot(_deg_to_rad(np.asarray(joint_deg)))
                wrist_cam.poll_frames()
                img = copy.deepcopy(wrist_cam.get_color_image())
                if img is None:
                    print(f"[capture] skipped unreadable frame at pose {idx}")
                    continue

                # Save raw image & one copy into compose folder for k=0
                cv2.imwrite(str(self.handeye_images_dir / f"wrist_image@pose_{idx}.png"), img)
                if idx == 0:
                    cv2.imwrite(str(self.compose_images_dir / f"wrist_image@pose_{idx}.png"), img)

                # Save EE pose
                rs_state = self.robot.get_state()
                ee_data = {
                    "translation": rs_state.cartesian_pos(),
                    "quat": rs_state.quat(),  # [x,y,z,w]
                }
                _write_json(self.handeye_images_dir / f"wrist_ee_pose_{idx}.json", ee_data)

                if self.show_gui:
                    cv2.imshow(f"Wrist image {idx}", img)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()

        finally:
            print("Returning robot to initial pose...")
            self.robot.move_and_wait_robot(_deg_to_rad(INIT_JOINT_DEG))
            wrist_cam.stop()

    # 2b) wrist-only hand–eye solve → write wristcam2ee, ee2base
    def handeye_calibration(self) -> Tuple[np.ndarray, np.ndarray]:
        extr = _read_json(self.extr_path)
        intr = _read_json(self.intr_path)

        wrist_images, _ = _get_images(self.handeye_images_dir / "wrist_image@pose_*.png")
        Rees, tees, _ = _get_ee_poses(self.handeye_images_dir / "wrist_ee_pose_*.json")
        if not wrist_images or not Rees:
            raise RuntimeError("handeye_calibration(): missing wrist images or EE poses.")

        if "wrist" not in intr:
            raise RuntimeError("handeye_calibration(): missing wrist intrinsics in intrinsics.json.")
        w = intr["wrist"]
        wrist_mtx = np.array([[w["fx"], 0, w["ppx"]], [0, w["fy"], w["ppy"]], [0, 0, 1]], dtype=float)

        R_chk2cam_list, t_chk2cam_list = [], []
        for idx, img in enumerate(wrist_images):
            if img is None:
                print(f"[handeye_calibration] skipped unreadable wrist image at index {idx}")
                Rees.pop(idx)
                tees.pop(idx)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cc, ids, mc, mid = DETECTOR.detectBoard(gray)
            if cc is None or len(cc) < 4:
                print(f"[handeye_calibration] not enough ChArUco corners in wrist image at index {idx}, skipping")
                Rees.pop(idx)# NOTE: not really in order
                tees.pop(idx)
                import pdb; pdb.set_trace()
                continue
            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                cc, ids, CALIB_BOARD, wrist_mtx, np.zeros(5), None, None
            )
            if not ok:
                print(f"[handeye_calibration] pose failure in wrist image at index {idx}, skipping")
                Rees.pop(idx)
                tees.pop(idx)
                continue
            R_chk, _ = cv2.Rodrigues(rvec)
            R_chk2cam_list.append(R_chk)
            t_chk2cam_list.append(tvec)

            # Optional visualization
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawFrameAxes(vis, wrist_mtx, np.zeros(5), rvec, tvec, 0.1)
            cv2.imwrite(str(self.marked_images_dir / f"handeye_calibration_{idx}.png"), vis)
            if self.show_gui:
                cv2.imshow(f"Hand-Eye {idx}", vis)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

        if not R_chk2cam_list:
            raise RuntimeError("handeye_calibration(): no valid ChArUco detections in wrist images.")

        T_wristcam2ee = calib.solve_EE_to_camera(Rees, tees, R_chk2cam_list, t_chk2cam_list)
        T_ee2base = calib.compose_transform(Rees[0], tees[0])

        extr["wristcam2ee"] = np.asarray(T_wristcam2ee, dtype=float).tolist()
        extr["ee2base"] = np.asarray(T_ee2base, dtype=float).tolist()
        _write_json(self.extr_path, extr)

        print("handeye_calibration(): wrote wristcam2ee and ee2base")

        return T_wristcam2ee, T_ee2base

    # 3) compose: cam2cam & cam2base (optimize when pairs exist; else fallback via board)
    def compose(self) -> None:
        extr = _read_json(self.extr_path)
        intr = _read_json(self.intr_path)
        board2cam: dict = extr.get("board2cam", {})
        if not board2cam:
            raise RuntimeError("compose(): board2cam not found; run fixed_calibration() first.")
        if "wristcam2ee" not in extr or "ee2base" not in extr:
            raise RuntimeError("compose(): wristcam2ee or ee2base missing; run handeye_calibration() first.")
        if "wrist" not in board2cam:
            raise RuntimeError("compose(): no camera named 'wrist' in board2cam; cannot identify wrist camera.")

        wrist_key = "wrist"
        cam_names = list(board2cam.keys())

        if wrist_key not in intr:
            raise RuntimeError("compose(): missing wrist intrinsics in intrinsics.json.")
        w = intr[wrist_key]
        wrist_mtx = np.array([[w["fx"], 0, w["ppx"]], [0, w["fy"], w["ppy"]], [0, 0, 1]], dtype=float)

        extr.setdefault("cam2cam", {})
        extr.setdefault("cam2base", {})

        T_wristcam2ee = np.asarray(extr["wristcam2ee"], dtype=float)
        T_ee2base = np.asarray(extr["ee2base"], dtype=float)

        # Always write wrist->base
        T_wrist2base = T_ee2base @ T_wristcam2ee
        extr["cam2base"][_pair_key(wrist_key, "base")] = T_wrist2base.tolist()

        # Fallback transforms via board frame
        T_board2wrist = np.asarray(board2cam[wrist_key], dtype=float)
        T_wrist2board = np.linalg.inv(T_board2wrist)

        for other in cam_names:
            if other == wrist_key:
                continue

            # Try optimization first if intrinsics & paired images exist
            other_mtx = None
            if other in intr:
                o = intr[other]
                other_mtx = np.array([[o["fx"], 0, o["ppx"]],
                                      [0, o["fy"], o["ppy"]],
                                      [0, 0, 1]], dtype=float)

            T_wrist2other: Optional[np.ndarray] = None
            if other_mtx is not None:
                T_wrist2other = _estimate_wrist2other_via_charuco(
                    images_dir=self.compose_images_dir,
                    other=other,
                    wrist_mtx=wrist_mtx,
                    other_mtx=other_mtx,
                    show_gui=self.show_gui,
                )

            # Fallback: compose via board if optimizer couldn’t run
            if T_wrist2other is None:
                T_board2other = np.asarray(board2cam[other], dtype=float)
                T_wrist2other = T_wrist2board @ T_board2other

            # Write cam2cam and cam2base
            extr["cam2cam"][_pair_key(wrist_key, other)] = T_wrist2other.tolist()
            T_other2base = T_ee2base @ T_wristcam2ee @ T_wrist2other
            extr["cam2base"][_pair_key(other, "base")] = T_other2base.tolist()

        _write_json(self.extr_path, extr)
        print("compose(): wrote cam2cam and cam2base")

    def close(self) -> None:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if self.robot:
            self.robot.stop()


# =================
# Simple CLI entry
# =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "calibration script for realsense cameras and xarm robot",
        description="Requires both a wrist-mounted and front-facing RealSense camera to be connected.",
    )
    parser.add_argument("work_dir", help="folder to save calibration info")
    parser.add_argument("--mode", default = "all", choices=["fixed", "handeye", "compose", "all"])
    parser.add_argument("--robot-ip", default="192.168.1.196", help="xArm robot IP")
    parser.add_argument("--show-gui", action="store_true", default=False)
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser()

    # Camera serial numbers — update these for your hardware setup.
    # Find serials by running:
    #   python -c "import pyrealsense2 as rs; ctx = rs.context(); [print(f'{d.get_info(rs.camera_info.name)}: {d.get_info(rs.camera_info.serial_number)}') for d in ctx.devices]"
    serials = {"wrist": "130322273767", "front": "239222300843"}

    # Hardware-reset all cameras to clear stale USB state before starting
    if args.mode in ("fixed", "handeye", "all"):
        import time
        print("Hardware-resetting RealSense devices...")
        ctx = rs.context()
        for dev in ctx.devices:
            serial = dev.get_info(rs.camera_info.serial_number)
            if serial in serials.values():
                name = dev.get_info(rs.camera_info.name)
                print(f"  Resetting {name} ({serial})...")
                dev.hardware_reset()
        del ctx
        time.sleep(3)

    if args.mode == "fixed":
        cal = HandEyeCalibrator(work_dir=work_dir, robot_ip=None, show_gui=args.show_gui, init_robot=False)
        cal.fixed_calibration(serials)
        cal.close()
    elif args.mode == "handeye":
        cal = HandEyeCalibrator(work_dir=work_dir, robot_ip=args.robot_ip, show_gui=args.show_gui, init_robot=True)
        cal.capture_wrist_images(serials.get("wrist"))
        cal.handeye_calibration()
        cal.close()
    elif args.mode == "compose":
        cal = HandEyeCalibrator(work_dir=work_dir, robot_ip=None, show_gui=args.show_gui, init_robot=False)
        cal.compose()
        cal.close()
    elif args.mode == "all":
        cal = HandEyeCalibrator(work_dir=work_dir, robot_ip=args.robot_ip, show_gui=args.show_gui, init_robot=True)
        cal.fixed_calibration(serials)
        cal.capture_wrist_images(serials.get("wrist"))
        cal.handeye_calibration()
        cal.compose()
        cal.close()
