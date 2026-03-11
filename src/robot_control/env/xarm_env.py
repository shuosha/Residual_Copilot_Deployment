"""Base xArm7 environment with shared infrastructure for all control modes.

Subclasses: TeleopEnv, PolicyEnv, ReplayEnv, SharedAutonomyEnv
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import os
import sys
import contextlib
import time
import threading
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from pathlib import Path
from multiprocessing.managers import SharedMemoryManager
from typing import Callable, Sequence, List, Literal, Optional, Union

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import multiprocess as mp

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout and stderr output."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

with suppress_stdout():
    import pygame
    pygame.init()

from robot_control.utils.utils import get_root, mkdir
from robot_control.perception.perception import Perception
from robot_control.control.xarm_controller import XarmController
from robot_control.agents.action_agent import ActionAgent
from robot_control.agents.teleop_agent import TeleopAgent
from robot_control.agents.shared_autonomy_agent import SharedAutonomyAgent
from robot_control.agents.policy_agent import PolicyAgent

_AGENT_MAP = {
    "gello": TeleopAgent,
    "keyboard": TeleopAgent,
    "residual": SharedAutonomyAgent,
    "residual_offline": SharedAutonomyAgent,
    "policy": PolicyAgent,
    "replay": PolicyAgent,
}
from robot_control.utils.kinematics_utils import (
    KinHelper, trans_mat_to_pos_quat, gripper_raw_to_qpos,
    pos_eef_to_fingertip, pos_fingertip_to_eef,
)
from robot_control.perception.camera.multi_realsense import MultiRealsense
from robot_control.perception.camera.single_realsense import SingleRealsense
from robot_control.utils.data_storage import (
    store_state_data as _store_state_data,
    store_robot_data as _store_robot_data,
)

root: Path = get_root(__file__)

# ---- Network Constants ----
XARM7_LEFT_IP = "192.168.1.196"
XARM7_RIGHT_IP = "192.168.1.224"

# ---- Control Constants ----
INIT_POSE_SEND_STEPS = 100        # steps to send init pose command

# ---- Image Processing Constants ----
DEPTH_VIZ_ALPHA = 0.03            # alpha for depth visualization colormap


class EnvEnum(Enum):
    """Debug verbosity levels for environment logging."""
    NONE = 0
    INFO = 1
    DEBUG = 2
    VERBOSE = 3


class XarmEnv(mp.Process):
    """Base environment for xArm7 robot control.

    Handles cameras, perception, xarm controller, action agent, image display,
    and process lifecycle. Subclasses implement run() for mode-specific behavior.
    """

    def __init__(
        self,
        # --------------------- Logging ---------------------
        debug: int = 0,
        exp_name: str = "recording",
        data_dir: Path = Path("data"),

        # --------------------- Cameras ---------------------
        realsense: Union[MultiRealsense, SingleRealsense, None] = None,
        shm_manager: Union[SharedMemoryManager, None] = None,
        serial_numbers: Union[Sequence[str], None] = None,
        resolution: tuple[int, int] = (848, 480),
        capture_fps: int = 30,
        record_fps: Union[int, None] = 0,
        record_time: Union[float, None] = 60 * 10,
        enable_depth: bool = True,
        enable_color: bool = True,

        # --------------------- Perception ---------------------
        perception: Union[Perception, None] = None,
        perception_process_func: Union[Callable, None] = None,

        # --------------------- Robot ---------------------
        control_mode: Literal["position_control", "velocity_control"] = "position_control",
        admittance_control: bool = False,
        ema_factor: float = 0.7,
        bimanual: bool = False,
        robot_ip: List[str] = [XARM7_LEFT_IP],
        gripper_enable: bool = False,

        # --------------------- Control ---------------------
        action_receiver: Literal["gello", "keyboard", "policy", "replay", "residual", "residual_offline"] = "gello",
        action_agent_fps: float = 10.0,
        init_poses: Union[List[np.ndarray], None] = [],
    ) -> None:
        super().__init__()

        # ------------ logging --------------
        self.debug = 0 if debug is None else (2 if debug is True else debug)
        self.exp_name = exp_name
        self.data_dir = Path(data_dir)

        # ------------ cameras --------------
        if realsense is not None:
            assert isinstance(realsense, (MultiRealsense, SingleRealsense))
            self.realsense = realsense
            self.serial_numbers = list(self.realsense.cameras.keys())
        else:
            self.realsense = MultiRealsense(
                shm_manager=shm_manager,
                serial_numbers=serial_numbers,
                resolution=resolution,
                capture_fps=capture_fps,
                enable_depth=enable_depth,
                enable_color=enable_color,
                process_depth=False,
                verbose=self.debug >= EnvEnum.VERBOSE.value,
            )
            self.serial_numbers = list(self.realsense.cameras.keys())

        self.realsense.set_exposure(exposure=None)
        self.realsense.set_white_balance(white_balance=None)

        self.capture_fps = capture_fps
        self.record_fps = record_fps

        # ------------ perception --------------
        if perception is not None:
            assert isinstance(perception, Perception)
            self.perception = perception
        else:
            self.perception = Perception(
                realsense=self.realsense,
                capture_fps=self.realsense.capture_fps,
                record_fps=record_fps,
                record_time=record_time,
                process_func=perception_process_func,
                exp_name=exp_name,
                data_dir=data_dir,
                verbose=self.debug >= EnvEnum.VERBOSE.value,
            )

        # ----------- robot ---------------
        self.bimanual = bimanual
        self.gripper_enable = gripper_enable

        if self.bimanual:
            assert len(robot_ip) == 2, "Bimanual xArm7 requires two robot IPs"
            self.left_xarm_controller = XarmController(
                start_time=time.time(), ip=robot_ip[0],
                gripper_enable=gripper_enable, control_mode=control_mode,
                admittance_control=admittance_control, ema_factor=ema_factor,
                comm_update_fps=action_agent_fps, robot_id=0,
                verbose=self.debug >= EnvEnum.VERBOSE.value,
            )
            self.right_xarm_controller = XarmController(
                start_time=time.time(), ip=robot_ip[1],
                gripper_enable=gripper_enable, control_mode=control_mode,
                admittance_control=admittance_control, ema_factor=ema_factor,
                comm_update_fps=action_agent_fps, robot_id=1,
                verbose=self.debug >= EnvEnum.VERBOSE.value,
            )
            self.xarm_controller = None
        else:
            assert len(robot_ip) == 1, "Single xArm7 requires one robot IP"
            self.xarm_controller = XarmController(
                start_time=time.time(), ip=robot_ip[0],
                gripper_enable=gripper_enable, control_mode=control_mode,
                admittance_control=admittance_control, ema_factor=ema_factor,
                comm_update_fps=action_agent_fps, robot_id=-1,
                verbose=self.debug >= EnvEnum.VERBOSE.value,
            )
            self.left_xarm_controller = None
            self.right_xarm_controller = None

        # ----------- action agent --------------
        self.action_receiver = action_receiver
        self.init_poses = init_poses

        agent_cls = _AGENT_MAP[action_receiver]
        self.action_agent = agent_cls(bimanual=self.bimanual, action_receiver=action_receiver)

        # ----------- process state --------------
        self.state = mp.Manager().dict()
        self._real_alive = mp.Value('b', False)
        self.start_time = 0
        mp.Process.__init__(self)
        self._alive = mp.Value('b', False)

        # ----------- image visualization --------------
        img_w, img_h = resolution
        views_per_cam = 2
        assert len(self.realsense.serial_numbers) > 0, "At least one Realsense camera must be connected"
        num_cams = max(len(self.realsense.serial_numbers), 1)

        unscaled_width = 2 * img_w * views_per_cam
        unscaled_height = 2 * img_h * num_cams

        screen_info = pygame.display.Info()
        max_screen_w, max_screen_h = screen_info.current_w, screen_info.current_h
        scale = min(1.0, max_screen_w / unscaled_width, max_screen_h / unscaled_height)

        self.screen_width = int(unscaled_width * scale)
        self.screen_height = int(unscaled_height * scale)

        self.image_data = mp.Array(
            'B',
            np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8).flatten(),
        )

    # ========== Process Control Methods ==========

    def real_start(self, start_time) -> None:
        """Start cameras, perception, robot controllers, and action agent."""
        self._real_alive.value = True
        print("starting real env")

        self.realsense.start()
        self.realsense.restart_put(start_time + 1)
        time.sleep(2)

        if self.perception is not None:
            self.perception.start()

        if self.bimanual:
            self.left_xarm_controller.start()
            self.right_xarm_controller.start()
        else:
            self.xarm_controller.start()

        self.action_agent.start()

        while not self.real_alive:
            self._real_alive.value = True
            print(".", end="")
            time.sleep(0.5)

        print("real env started")

        self.update_real_state_t = threading.Thread(name="update_real_state", target=self.update_real_state)
        self.update_real_state_t.start()

    def real_stop(self, wait=False) -> None:
        """Stop robot controllers, perception, cameras, and cleanup threads."""
        self._real_alive.value = False
        if self.bimanual:
            if self.left_xarm_controller.is_controller_alive:
                self.left_xarm_controller.stop()
            if self.right_xarm_controller.is_controller_alive:
                self.right_xarm_controller.stop()
        else:
            if self.xarm_controller.is_controller_alive:
                self.xarm_controller.stop()
        if self.perception is not None and self.perception.alive.value:
            self.perception.stop()
        self.realsense.stop(wait=False)

        self.image_display_thread.join()
        self.update_real_state_t.join()

        print("======= Real Env Stopped =======")

    @property
    def real_alive(self) -> bool:
        """Check if real environment is alive (perception, robot controllers all running)."""
        alive = self._real_alive.value
        if self.perception is not None:
            alive = alive and self.perception.alive.value
        controller_alive = (
            (self.bimanual and self.left_xarm_controller.is_controller_alive
             and self.right_xarm_controller.is_controller_alive)
            or (not self.bimanual and self.xarm_controller.is_controller_alive)
        )
        alive = alive and controller_alive
        self._real_alive.value = alive
        return self._real_alive.value

    # ========== State Update Methods ==========

    def _update_perception(self) -> None:
        if self.perception.alive.value:
            if not self.perception.perception_q.empty():
                self.state["perception_out"] = {
                    "value": self.perception.perception_q.get()
                }

    def _update_robot(self) -> None:
        if self.bimanual:
            if self.left_xarm_controller.is_controller_alive and self.right_xarm_controller.is_controller_alive:
                self.state["trans_out"] = {
                    "capture_time": self.left_xarm_controller.cur_time_q.value,
                    "left_value": np.array(self.left_xarm_controller.cur_trans_q[:]).reshape(4, 4),
                    "right_value": np.array(self.right_xarm_controller.cur_trans_q[:]).reshape(4, 4),
                }
                self.state["qpos_out"] = {
                    "left_value": np.array(self.left_xarm_controller.cur_qpos_q[:]),
                    "right_value": np.array(self.right_xarm_controller.cur_qpos_q[:]),
                }
                self.state["gripper_out"] = {
                    "left_value": np.array(self.left_xarm_controller.cur_gripper_q[:]),
                    "right_value": np.array(self.right_xarm_controller.cur_gripper_q[:]),
                }
        else:
            if self.xarm_controller.is_controller_alive:
                self.state["trans_out"] = {
                    "capture_time": self.xarm_controller.cur_time_q.value,
                    "value": np.array(self.xarm_controller.cur_trans_q[:]).reshape(4, 4),
                }
                self.state["qpos_out"] = {
                    "value": np.array(self.xarm_controller.cur_qpos_q[:]),
                }
                self.state["gripper_out"] = {
                    "value": np.array(self.xarm_controller.cur_gripper_q[:]),
                }
                self.state["force_out"] = {
                    "value": np.array(self.xarm_controller.cur_force_q[:]),
                }

    def _update_command(self) -> None:
        if self.bimanual:
            raise NotImplementedError("Bimanual command update is not implemented yet")
        else:
            if self.action_agent.is_alive():
                self.state["action_qpos_out"] = {
                    "capture_time": self.action_agent.cur_time_q.value,
                    "value": np.array(self.action_agent.cur_qpos_comm[:7]),
                }
                self.state["action_trans_out"] = {
                    "value": np.array(self.action_agent.cur_eef_trans[:]).reshape(4, 4),
                }
                self.state["action_gripper_out"] = {
                    "value": np.array(self.action_agent.cur_qpos_comm[7:]),
                }

    def update_real_state(self) -> None:
        """Main state update loop running in separate thread."""
        while self.real_alive:
            try:
                self._update_robot()
                if self.perception is not None:
                    self._update_perception()
                if self.action_agent is not None:
                    self._update_command()
            except BaseException as e:
                print(f"Error in update_real_state: {e.with_traceback()}")
                break
        print("update_real_state stopped")

    # ========== Visualization Methods ==========

    def display_image(self):
        """Display camera images in pygame window, updating from shared memory buffer."""
        self.image_window = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Image Display Window')
        while self._alive.value:
            image = np.frombuffer(self.image_data.get_obj(), dtype=np.uint8).reshape(
                (self.screen_height, self.screen_width, 3)
            )
            pygame_image = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            self.image_window.blit(pygame_image, (0, 0))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Image display window stopped")
                    self.stop()
                    pygame.quit()
                    return

            time.sleep(1 / self.realsense.capture_fps)
        print("Image display stopped")

    def start_image_display(self):
        """Start image display thread."""
        self.image_display_thread = threading.Thread(name="display_image", target=self.display_image)
        self.image_display_thread.start()

    def _build_display_frame(self, rgbs, depths):
        """Compose RGB+depth grid, resize, and write to shared memory."""
        row_imgs = []
        for row in range(len(self.realsense.serial_numbers)):
            rgb = cv2.cvtColor(rgbs[row], cv2.COLOR_BGR2RGB)
            depth = cv2.applyColorMap(
                cv2.convertScaleAbs(depths[row], alpha=DEPTH_VIZ_ALPHA),
                cv2.COLORMAP_JET,
            )
            row_imgs.append(np.hstack((rgb, depth)))
        combined_img = np.vstack(row_imgs)

        combined_img = cv2.resize(
            combined_img,
            (self.screen_width, self.screen_height),
            interpolation=cv2.INTER_AREA,
        )

        np.copyto(
            np.frombuffer(self.image_data.get_obj(), dtype=np.uint8).reshape(
                (self.screen_height, self.screen_width, 3)
            ),
            combined_img,
        )

    # ========== Control and Action Methods ==========

    def get_qpos_from_action_8d(self, action, curr_qpos) -> np.ndarray:
        """Convert 8D action (pos, quat, gripper) to joint positions via IK."""
        assert action.shape[0] == 8, "Action shape must be (8,) for robot control"
        action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action

        pos = action_np[0:3]
        quat_wxyz = action_np[3:7]
        quat_xyzw = np.roll(quat_wxyz, -1)
        gripper_qpos = action_np[7:]

        tf = np.eye(4)
        tf[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        tf[:3, 3] = pos
        goal_qpos = self.kin_helper.compute_ik_sapien(initial_qpos=curr_qpos, tf=tf)

        return np.concatenate([goal_qpos, gripper_qpos], axis=0)

    def set_robot_initial_pose(self, init_pose: Sequence[float], fps=15.0) -> None:
        """Set robot to initial pose (degrees → radians)."""
        if self.bimanual:
            raise NotImplementedError("Bimanual replay is not implemented yet")
        else:
            print("Resetting robot to initial pose (deg):", init_pose)
            init_pose *= np.pi / 180
            print("Initial pose (radians):", init_pose)
            init_pose = init_pose.tolist()

            assert self.alive, "Environment must be running to set initial pose"
            self.xarm_controller.teleop_activated.value = True
            for _ in range(INIT_POSE_SEND_STEPS):
                tic = time.time()
                self.action_agent.command[:] = init_pose
                time.sleep(max(0, 1 / fps - (time.time() - tic)))
            time.sleep(1)
            print("Initial pose set")

            if self.action_receiver in ("replay", "policy"):
                self.action_agent.record_start.value = True
            elif self.action_receiver in ("keyboard", "gello", "residual"):
                self.xarm_controller.teleop_activated.value = False

    # ========== Data Storage Methods ==========

    def store_robot_data(self, trans_out, qpos_out, gripper_out, action_qpos_out,
                         action_trans_out, action_gripper_out, robot_obs_record_dir,
                         robot_action_record_dir, force_out=None):
        """Store teleop-mode robot data. Delegates to data_storage module."""
        _store_robot_data(trans_out, qpos_out, gripper_out, action_qpos_out,
                          action_trans_out, action_gripper_out, robot_obs_record_dir,
                          robot_action_record_dir, force_out=force_out,
                          bimanual=self.bimanual, gripper_enable=self.gripper_enable)

    # ========== Shared Run Helpers ==========

    def _init_run(self):
        """Common initialization at the start of run(). Returns (robot_obs_dir, robot_action_dir, rgbs, depths, fps)."""
        robot_obs_record_dir = root / "logs" / self.data_dir / self.exp_name / "robot_obs"
        os.makedirs(robot_obs_record_dir, exist_ok=True)

        robot_action_record_dir = root / "logs" / self.data_dir / self.exp_name / "robot_action"
        os.makedirs(robot_action_record_dir, exist_ok=True)

        self.kin_helper = KinHelper(robot_name='xarm7')

        rgbs = []
        depths = []
        resolution = self.realsense.resolution
        for _ in range(len(self.realsense.serial_numbers)):
            rgbs.append(np.zeros((resolution[1], resolution[0], 3), np.uint8))
            depths.append(np.zeros((resolution[1], resolution[0]), np.uint16))

        fps = self.record_fps if self.record_fps > 0 else self.realsense.capture_fps

        return robot_obs_record_dir, robot_action_record_dir, rgbs, depths, fps

    def _read_state(self):
        """Read current state snapshot. Returns deepcopy of self.state."""
        return deepcopy(self.state)

    def _handle_recording_signals(self, mode="teleop"):
        """Handle recording start/stop signals from action agent."""
        if self.action_agent.record_start.value:
            self.perception.set_record_start()
            self.action_agent.record_start.value = False

        if self.action_agent.record_stop.value:
            if self.action_agent.record_failed.value:
                self.perception.set_record_failed(mode=mode)
            self.perception.set_record_stop(mode=mode)
            self.action_agent.record_stop.value = False

    def _update_images(self, perception_out, rgbs, depths):
        """Update RGB and depth images from perception output."""
        if perception_out is not None:
            for k, v in perception_out['value'].items():
                rgbs[k] = v["color"]
                depths[k] = v["depth"]

    # ========== Lifecycle ==========

    @abstractmethod
    def run(self) -> None:
        """Main process loop. Must be implemented by subclasses."""
        ...

    def get_intrinsics(self):
        """Get camera intrinsics from RealSense."""
        return self.realsense.get_intrinsics()

    @property
    def alive(self) -> bool:
        alive = self._alive.value and self.real_alive
        self._alive.value = alive
        return alive

    def start(self) -> None:
        self.start_time = time.time()
        self._alive.value = True
        self.real_start(time.time())
        self.start_image_display()
        super().start()

    def stop(self) -> None:
        self._alive.value = False
        self.real_stop()
