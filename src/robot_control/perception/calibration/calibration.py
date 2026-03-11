import glob
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import cv2
# choose a dictionary and Charuco board size that matches your printed board
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
# squaresX, squaresY = number of chessboard squares in X and Y
# squareLength = size of each square side (in meters)
# markerLength = size of the ArUco markers inside each square
CALIB_BOARD = cv2.aruco.CharucoBoard(
    size=(4, 5),
    squareLength=0.048,
    markerLength=0.036,
    dictionary=ARUCO_DICT,
)
DETECTOR_PARAMS = cv2.aruco.CharucoParameters()
DETECTOR = cv2.aruco.CharucoDetector(
    CALIB_BOARD,
    DETECTOR_PARAMS,
)

def decompose_transform(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def compose_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def save_camera_intrinsics(intrinsics, save_as):
    intrinsic_dict = {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.ppx,
        "cy": intrinsics.ppy,
        "distortion model": str(intrinsics.model),
        "coeffs": intrinsics.coeffs,
        "height": intrinsics.height,
        "width": intrinsics.width,
    }
    with open(save_as, "w") as f:
        json.dump(intrinsic_dict, f, indent=4)
    print(f"Camera intrinsics saved to {save_as}")

def load_camera_intrinsic(file_path: Path):
    with open(file_path, "r") as f:
        intr = json.load(f)

    mtx = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    coeff = np.array(intr["coeffs"])

    return mtx, coeff

def get_image_paths(data_dir: Path, img_prefix: str) -> list[str]:
    path_prefix = data_dir / img_prefix
    img_names = sorted(glob.glob(str(path_prefix)))

    return img_names


def get_images(data_dir: Path, img_prefix) -> tuple[list[cv2.typing.MatLike], list[str]]:
    img_names = get_image_paths(data_dir, img_prefix)
    images = [cv2.imread(img_name, 1) for img_name in img_names]

    return images, img_names


def get_ee_data_names(data_dir: Path, ee_data_prefix: str) -> list[str]:
    path_prefix = data_dir / ee_data_prefix
    ee_data_names = sorted(glob.glob(str(path_prefix)))

    return ee_data_names


def get_ee_poses(data_dir: Path, ee_data_prefix: str) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    ee_data_names = get_ee_data_names(data_dir, ee_data_prefix)

    rots = []
    transls = []

    for name in ee_data_names:
        with open(name) as f:
            ee_transform = json.load(f)

        translation = np.array(ee_transform["translation"])

        # (rx, ry, rz, rw)
        quat = ee_transform["quat"]
        rot_mat = Rotation.from_quat(quat, scalar_first=False).as_matrix()

        rots.append(rot_mat)
        transls.append(translation)

    return rots, transls, ee_data_names


def get_charucoboard_corners(
    color_img: cv2.typing.MatLike, 
    detector: cv2.aruco.CharucoDetector = DETECTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (corners, ids) via detectMarkers.
    """
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    detected_corners, detected_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if marker_ids is None or len(marker_ids) == 0:
        raise RuntimeError("No ArUco markers found.")

    return detected_corners, detected_ids, marker_corners, marker_ids

def get_charucoboard_pose(
    detected_corners: np.ndarray,
    detected_ids: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    board: cv2.aruco.CharucoBoard,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (R_cam_to_board, t_cam_to_board) via estimatePoseCharucoBoard.
    """

    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            detected_corners,
            detected_ids,
            board,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            rvec=None,
            tvec=None,
        )
    
    if not retval:    
        raise ValueError("pose estimation failed")

    return rvec, tvec


def get_checkerboard_corners(
    color_img: cv2.typing.MatLike, checker_rows: int, checker_columns: int, visualizing_image_scale: float
) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    # may need to tune
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # make gray image
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # flag check
    # ret, corners = cv2.findChessboardCorners(
    #     gray_img, (checker_rows, checker_columns), flags=cv2.CALIB_CB_ADAPTIVE_THRESH
    # )
    ret, corners = cv2.findChessboardCorners(gray_img, (checker_rows, checker_columns))
    # print(ret)
    corners_refined = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
    # corners_refined = corners
    cv2.drawChessboardCorners(color_img, (checker_rows, checker_columns), corners, ret)

    # for visualization, make a scaled image
    height, width, channels = color_img.shape
    scaled_img = cv2.resize(color_img, (int(width * visualizing_image_scale), int(height * visualizing_image_scale)))

    return scaled_img, corners_refined

def get_checker_object_points(checker_rows: int, checker_columns: int, square_size: float) -> np.ndarray:
    objp = np.zeros((checker_rows * checker_columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker_rows, 0:checker_columns].T.reshape(-1, 2)
    # objp[:, :2] = np.indices([checker_rows, checker_cols]).T.reshape(-1, 2)
    objp *= square_size

    return objp


def get_transform_from_cam_to_checker(
    checker_rows: int,
    checker_cols: int,
    square_size,
    corners: cv2.typing.MatLike,
    mtx: np.ndarray,
    dist_coeff: np.ndarray,
) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    objp = get_checker_object_points(checker_rows, checker_cols, square_size)
    ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist_coeff)

    R, _ = cv2.Rodrigues(rvec)

    return R, tvec

def solve_EE_to_camera(all_R_BaseToEE, all_t_BaseToEE, all_R_CamToCheck, all_t_CamToCheck) -> np.ndarray:
    R_gripper2base = all_R_BaseToEE
    t_gripper2base = all_t_BaseToEE
    R_target2cam = all_R_CamToCheck
    t_target2cam = all_t_CamToCheck

    # Solve: A X = X B
    R, t = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_TSAI
    )

    # Build 4x4 transform
    T = compose_transform(R, t)
    # T = np.eye(4)
    # T[:3, :3] = R
    # T[:3, 3] = t.flatten()
    return T