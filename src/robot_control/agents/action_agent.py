"""ActionAgent base class — shared infrastructure for all action agent subclasses.

Subclasses:
- TeleopAgent: gello/keyboard teleop
- SharedAutonomyAgent: residual online + offline
- PolicyAgent: policy rollout + replay
"""

from abc import abstractmethod
import multiprocess as mp
import time
import numpy as np
from pynput import keyboard

from robot_control.utils.udp_util import udpSender
from robot_control.utils.kinematics_utils import KinHelper

from robot_control.control.common.communication import XARM_CONTROL_PORT, XARM_CONTROL_PORT_L, XARM_CONTROL_PORT_R


class ActionAgent(mp.Process):

    name = "action_agent"
    kin_helper = KinHelper(robot_name='xarm7')

    _SPECIAL_KEY_MAP = {
        keyboard.Key.up: "up",
        keyboard.Key.down: "down",
        keyboard.Key.left: "left",
        keyboard.Key.right: "right",
        keyboard.Key.shift: "shift",
        keyboard.Key.shift_l: "shift",
        keyboard.Key.shift_r: "shift",
    }

    def __init__(
        self,
        bimanual=False,
    ) -> None:
        super().__init__()
        self.bimanual = bimanual
        if self.bimanual:
            self.log("Using bimanual joint mapping mode")

        # Base key states — recording controls shared by all agents
        self.key_states = {
            ",": False,
            ".": False,
            "/": False,
        }

        # Recording state
        self.record_start = mp.Value('b', False)
        self.record_stop = mp.Value('b', False)
        self.record_failed = mp.Value('b', False)

        # Shared command arrays
        if self.bimanual:
            self.command = mp.Array('d', [0.0] * 16)
            self.cur_qpos_comm = mp.Array('d', [0.0] * 16)
            self.cur_eef_trans = mp.Array('d', [0.0] * 32)
        else:
            self.command = mp.Array('d', [0.0] * 8)
            self.cur_qpos_comm = mp.Array('d', [0.0] * 8)
            self.cur_eef_trans = mp.Array('d', [0.0] * 16)

        self.cur_time_q = mp.Value('d', 0.0)
        self._alive = mp.Value('b', True)
        self.controller_quit = True

        self.command_sender = None
        self.command_sender_left = None
        self.command_sender_right = None

    def log(self, msg):
        """Print a log message with blue color formatting."""
        print(f"\033[94m{msg}\033[0m")

    def on_press(self, key):
        """Handle keyboard key press events and update key states."""
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = True
        except AttributeError:
            mapped = self._SPECIAL_KEY_MAP.get(key)
            if mapped and mapped in self.key_states:
                self.key_states[mapped] = True

    def on_release(self, key):
        """Handle keyboard key release events. Returns False on ESC to stop listener."""
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = False
        except AttributeError:
            if key == keyboard.Key.esc:
                return False
            mapped = self._SPECIAL_KEY_MAP.get(key)
            if mapped and mapped in self.key_states:
                self.key_states[mapped] = False

    def _handle_record_keys(self):
        """Process recording key states (shared by all agents)."""
        if self.key_states[","]:
            self.record_start.value = True
            self.record_failed.value = False
            self.log("Record start")
            time.sleep(0.2)

        if self.key_states["."]:
            self.record_stop.value = True
            self.log("Record stop and success")
            time.sleep(0.2)

        if self.key_states["/"]:
            self.record_stop.value = True
            self.record_failed.value = True
            self.log("Record stop and failed")
            time.sleep(0.2)

    @abstractmethod
    def _init_listeners(self):
        """Initialize mode-specific listeners (gello, keyboard, etc.)."""

    @abstractmethod
    def _process_command(self) -> list:
        """Process inputs and return command list to send. Return [] to skip sending."""

    @abstractmethod
    def _cleanup_listeners(self):
        """Clean up mode-specific listeners."""

    def run(self) -> None:
        """Main process loop: initialize listeners, process commands, send via UDP."""
        self._init_listeners()

        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_listener.start()

        if self.bimanual:
            self.command_sender_left = udpSender(port=XARM_CONTROL_PORT_L)
            self.command_sender_right = udpSender(port=XARM_CONTROL_PORT_R)
        else:
            self.command_sender = udpSender(port=XARM_CONTROL_PORT)

        time.sleep(1)
        self.log(f"ActionAgent start! (mode: {self.__class__.__name__})")

        while self.alive:
            try:
                current_time = time.time()
                command = self._process_command()

                if command:
                    if self.bimanual:
                        self.command_sender_left.send([command[0:8]])
                        self.command_sender_right.send([command[8:16]])
                    else:
                        self.command_sender.send([command])

                # data storage
                command_np = np.array(self.command[:])
                self.cur_time_q.value = current_time
                self.cur_qpos_comm[:] = command_np.tolist()

                fk = self.kin_helper.compute_fk_sapien_links(command_np[:7], [self.kin_helper.sapien_eef_idx])[0]
                self.cur_eef_trans[:] = fk.flatten()

            except Exception as e:
                print(f"Error in ActionAgent", e.with_traceback())
                break

        self.stop()
        if self.bimanual:
            self.command_sender_left.close()
            self.command_sender_right.close()
        else:
            self.command_sender.close()

        self._cleanup_listeners()
        self.keyboard_listener.stop()
        self.log(f"{'='*20} ActionAgent exit!")

    @property
    def alive(self):
        """Check if the action agent process is still alive."""
        alive = self._alive.value
        self._alive.value = alive
        return alive

    def stop(self, stop_controller=False):
        """Stop the action agent process and optionally send quit command to robot controllers."""
        if stop_controller:
            self.log("teleop stop controller")
            if self.command_sender is not None:
                self.command_sender.send(["quit"])
            if self.command_sender_left is not None:
                self.command_sender_left.send(["quit"])
            if self.command_sender_right is not None:
                self.command_sender_right.send(["quit"])
            time.sleep(1)
        self._alive.value = False
        self.log("teleop stop")
