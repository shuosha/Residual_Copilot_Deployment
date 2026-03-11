"""TeleopAgent — gello and keyboard teleop action agent."""

import time
import numpy as np
import transforms3d

from robot_control.agents.action_agent import ActionAgent
from robot_control.agents.gello_listener import GelloListener


class TeleopAgent(ActionAgent):

    # Keyboard teleop Cartesian step sizes (meters / radians per tick)
    KB_POS_STEP = 0.005        # 5 mm
    KB_POS_STEP_SLOW = 0.001   # 1 mm
    KB_ROT_STEP = np.deg2rad(1.0)       # 1 degree
    KB_ROT_STEP_SLOW = np.deg2rad(0.2)  # 0.2 degrees

    def __init__(
        self,
        action_receiver="gello",
        **kwargs,
    ) -> None:
        assert action_receiver in ("gello", "keyboard"), f"TeleopAgent only supports gello/keyboard, got {action_receiver}"
        super().__init__(**kwargs)
        self.action_receiver = action_receiver
        self.pause = False
        self.gello_listener = None
        self._kb_gripper = 0.0

        # Extend key states for teleop
        self.key_states.update({
            "p": False,
            # keyboard teleop keys
            "w": False, "a": False, "s": False, "d": False,
            "up": False, "down": False,
            "i": False, "k": False,
            "j": False, "l": False,
            "u": False, "o": False,
            "[": False, "]": False,
            "left": False, "right": False,
            "shift": False,
        })

    def _init_listeners(self):
        if self.action_receiver == "gello":
            gello_port = '/dev/ttyUSB0'
            baudrate = 57600
            self.gello_listener = GelloListener(
                bimanual=self.bimanual,
                gello_port=gello_port,
                baudrate=baudrate,
                bimanual_gello_port=['/dev/ttyUSB0', '/dev/ttyUSB1'],
            )
            self.log(f"initializing dynamixel gello listener with port: {gello_port} and baudrate: {baudrate}")
            self.gello_listener.start()
        else:
            self.log("Using keyboard Cartesian teleop mode")

    def _process_command(self) -> list:
        if self.key_states["p"]:
            self.pause = not self.pause
            self.log(f"teleop pause status: {self.pause}")
            time.sleep(0.2)

        self._handle_record_keys()

        if self.action_receiver == "gello":
            self._get_gello_command()
        else:
            self._get_keyboard_command()

        if self.pause:
            return []
        return list(self.command)

    def _get_gello_command(self):
        """Read gello joints and write to command buffer."""
        command_joints = self.gello_listener.get()
        assert command_joints.shape[0] in [8, 16], f"gello command shape should be (8,) or (16,), got {command_joints.shape}"

        if not self.pause:
            self.command[:] = command_joints.tolist()

    def _get_keyboard_command(self):
        """Generate joint commands from keyboard Cartesian increments via FK -> delta -> IK."""
        if self.pause:
            return

        # Get current joint state and compute FK
        cur_joints = np.array(self.command[:8])
        fk = self.kin_helper.compute_fk_sapien_links(cur_joints[:7], [self.kin_helper.sapien_eef_idx])[0]

        pos = fk[:3, 3].copy()
        R = fk[:3, :3].copy()

        # Select step sizes based on shift modifier
        pos_step = self.KB_POS_STEP_SLOW if self.key_states["shift"] else self.KB_POS_STEP
        rot_step = self.KB_ROT_STEP_SLOW if self.key_states["shift"] else self.KB_ROT_STEP

        # Apply position increments
        if self.key_states["w"]:
            pos[0] += pos_step
        if self.key_states["s"]:
            pos[0] -= pos_step
        if self.key_states["a"]:
            pos[1] += pos_step
        if self.key_states["d"]:
            pos[1] -= pos_step
        if self.key_states["up"]:
            pos[2] += pos_step
        if self.key_states["down"]:
            pos[2] -= pos_step

        # Apply rotation increments (local frame rotations)
        dr = np.zeros(3)
        if self.key_states["j"]:
            dr[0] += rot_step   # roll +
        if self.key_states["l"]:
            dr[0] -= rot_step   # roll -
        if self.key_states["i"]:
            dr[1] += rot_step   # pitch +
        if self.key_states["k"]:
            dr[1] -= rot_step   # pitch -
        if self.key_states["u"]:
            dr[2] += rot_step   # yaw +
        if self.key_states["o"]:
            dr[2] -= rot_step   # yaw -

        if np.any(dr != 0):
            dR = transforms3d.euler.euler2mat(dr[0], dr[1], dr[2], axes='sxyz')
            R = R @ dR

        # Handle gripper
        gripper_step = 0.05
        if self.key_states["["]:
            self._kb_gripper = max(0.0, self._kb_gripper - gripper_step)
        if self.key_states["]"]:
            self._kb_gripper = min(1.0, self._kb_gripper + gripper_step)
        if self.key_states["left"]:
            self._kb_gripper = 0.0
        if self.key_states["right"]:
            self._kb_gripper = 1.0

        # Build target transform and IK
        target_fk = np.eye(4)
        target_fk[:3, :3] = R
        target_fk[:3, 3] = pos

        next_joints = self.kin_helper.compute_ik_sapien(cur_joints[:7], target_fk, verbose=False)
        next_joints = next_joints.tolist()
        next_joints.append(self._kb_gripper)

        self.command[:] = next_joints

    def enforce_z_down(self, fk: np.ndarray) -> np.ndarray:
        """Modify forward kinematics transform to enforce z-axis pointing down while preserving yaw."""
        fk = fk.copy()
        R = fk[:3, :3]

        z_axis = np.array([0.0, 0.0, -1.0])
        x_curr = R[:, 0]
        x_proj = x_curr - np.dot(x_curr, z_axis) * z_axis
        norm = np.linalg.norm(x_proj)
        if norm < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = x_proj / norm

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        R_new = np.column_stack([x_axis, y_axis, z_axis])
        fk[:3, :3] = R_new
        return fk

    def _cleanup_listeners(self):
        if self.gello_listener is not None:
            self.gello_listener.stop()
