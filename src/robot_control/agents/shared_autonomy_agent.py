"""SharedAutonomyAgent — residual online (gello + residual) and offline action agent."""

import time
import multiprocess as mp

from robot_control.agents.action_agent import ActionAgent
from robot_control.agents.gello_listener import GelloListener


class SharedAutonomyAgent(ActionAgent):

    def __init__(
        self,
        action_receiver="residual",
        **kwargs,
    ) -> None:
        assert action_receiver in ("residual", "residual_offline"), \
            f"SharedAutonomyAgent only supports residual/residual_offline, got {action_receiver}"
        super().__init__(**kwargs)
        self.action_receiver = action_receiver
        self.offline = (action_receiver == "residual_offline")
        self.pause = False
        self.gello_listener = None

        # Shared autonomy mode flags
        self.use_residual_copilot = mp.Value('b', False)
        self.use_residual_bc = mp.Value('b', False)
        self.use_teleop = mp.Value('b', True)
        self.switching_controller = mp.Value('b', False)
        self.command_with_residual = mp.Array('d', [0.0] * 8)
        self.track_obj = mp.Value('b', False)
        self.reset = mp.Value('b', False)

        # Extend key states
        self.key_states.update({
            "p": False,
            "1": False,  # teleop mode
            "2": False,  # residual copilot mode
            "3": False,  # residual bc mode
            "m": False,  # retrack objects
            "r": False,  # return to initial pose (offline)
            "s": False,  # start inference (offline)
        })

    def _init_listeners(self):
        if not self.offline:
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

    def _process_command(self) -> list:
        # Pause
        if self.key_states["p"]:
            self.pause = not self.pause
            self.log(f"teleop pause status: {self.pause}")
            time.sleep(0.2)

        # Recording
        self._handle_record_keys()

        # Mode switching
        if self.key_states["1"]:
            self.use_teleop.value = True
            self.use_residual_copilot.value = False
            self.use_residual_bc.value = False
            self.log("control mode: teleop")
            time.sleep(0.2)
        if self.key_states["2"]:
            self.use_teleop.value = False
            self.use_residual_copilot.value = True
            self.use_residual_bc.value = False
            self.log("control mode: residual copilot")
            time.sleep(0.2)
        if self.key_states["3"]:
            self.use_teleop.value = False
            self.use_residual_copilot.value = False
            self.use_residual_bc.value = True
            self.log("control mode: residual bc")
            time.sleep(0.2)

        # Object retracking
        if self.key_states["m"]:
            self.track_obj.value = True
            self.log(f"retrack objects: {self.track_obj.value}")
            time.sleep(0.2)

        # Offline: reset to initial pose / start inference
        if self.offline:
            if self.key_states["r"]:
                self.reset.value = True
                self.record_stop.value = True
                self.log("offline: returning to initial pose")
                time.sleep(0.2)
            if self.key_states["s"]:
                self.reset.value = False
                self.record_start.value = True
                self.log("offline: starting inference")
                time.sleep(0.2)

        if self.gello_listener is not None:
            # Online mode: read gello
            command_joints = self.gello_listener.get()
            assert command_joints.shape[0] in [8, 16], \
                f"gello command shape should be (8,) or (16,), got {command_joints.shape}"

            if not self.pause:
                self.command[:] = command_joints.tolist()

            if self.pause:
                return []

            # If using residual or resetting, use residual-modified command
            if not self.switching_controller.value:
                if self.use_residual_copilot.value or self.use_residual_bc.value or self.reset.value:
                    return list(self.command_with_residual)
                return list(self.command)
            return []
        else:
            # Offline mode: main process writes commands directly
            return list(self.command)

    def _cleanup_listeners(self):
        if self.gello_listener is not None:
            self.gello_listener.stop()
