"""Teleop environment — human controls robot via Gello or keyboard."""

import time
from copy import deepcopy

from robot_control.env.xarm_env import XarmEnv


class TeleopEnv(XarmEnv):
    """Teleop-only environment. No policy, no init pose, no timestep tracking."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self) -> None:
        robot_obs_dir, robot_action_dir, rgbs, depths, fps = self._init_run()

        while self.alive:
            try:
                tic = time.time()
                state = self._read_state()

                mode = "teleop"
                self._handle_recording_signals(mode=mode)

                perception_out = state.get("perception_out", None)
                trans_out = state.get("trans_out", None)
                qpos_out = state.get("qpos_out", None)
                gripper_out = state.get("gripper_out", None)
                force_out = state.get("force_out", None)
                action_qpos_out = state.get("action_qpos_out", None)
                action_trans_out = state.get("action_trans_out", None)
                action_gripper_out = state.get("action_gripper_out", None)

                self._update_images(perception_out, rgbs, depths)

                # Store data
                if trans_out is not None and action_qpos_out is not None:
                    self.store_robot_data(
                        trans_out, qpos_out, gripper_out,
                        action_qpos_out, action_trans_out, action_gripper_out,
                        robot_obs_dir, robot_action_dir, force_out,
                    )

                self._build_display_frame(rgbs, depths)

                time.sleep(max(0, 1 / fps - (time.time() - tic)))
                if 1 / (time.time() - tic) > fps + 1 or 1 / (time.time() - tic) < fps - 1:
                    print("real env fps: ", 1 / (time.time() - tic), f"(target: {fps})")

            except BaseException as e:
                print(f"Error in TeleopEnv: {e.with_traceback()}")
                break

        self.action_agent.stop()
        self.stop()
        print("TeleopEnv process stopped")
