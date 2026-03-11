"""Replay environment — replays pre-recorded action trajectories."""

import time
from typing import List

import numpy as np

from robot_control.env.xarm_env import XarmEnv


class ReplayEnv(XarmEnv):
    """Replays pre-recorded cartesian action trajectories on the robot."""

    def __init__(self, action_trajs: List[np.ndarray], **kwargs):
        super().__init__(action_receiver="replay", **kwargs)
        self.action_trajs = action_trajs
        self.total_trajs = len(action_trajs)
        print(f"Replaying {self.total_trajs} trajectories")

    def run(self) -> None:
        robot_obs_dir, robot_action_dir, rgbs, depths, fps = self._init_run()

        timestep = 0
        eps_idx = 0

        # Set initial pose
        if len(self.init_poses) == 1:
            init_pose = self.init_poses[0].copy()
        else:
            init_pose = self.init_poses[eps_idx % len(self.init_poses)].copy()
        self.set_robot_initial_pose(init_pose, fps=fps)
        print(f"Robot initial pose set for episode {eps_idx}")

        total_timesteps = self.action_trajs[eps_idx].shape[0]
        time.sleep(1)

        while self.alive:
            try:
                tic = time.time()
                state = self._read_state()

                self._handle_recording_signals(mode="teleop")

                perception_out = state.get("perception_out", None)
                trans_out = state.get("trans_out", None)
                qpos_out = state.get("qpos_out", None)
                gripper_out = state.get("gripper_out", None)
                force_out = state.get("force_out", None)
                action_qpos_out = state.get("action_qpos_out", None)
                action_trans_out = state.get("action_trans_out", None)
                action_gripper_out = state.get("action_gripper_out", None)

                self._update_images(perception_out, rgbs, depths)

                # Replay step
                if timestep == 0:
                    print(f"Starting episode {eps_idx} with {total_timesteps} timesteps")
                if timestep < total_timesteps:
                    action = self.action_trajs[eps_idx][timestep]
                    command = self.get_qpos_from_action_8d(action, qpos_out["value"])
                    self.action_agent.command[:] = command
                    timestep += 1
                else:
                    print(f"Episode {eps_idx} finished after {timestep} timesteps")
                    self.perception.set_record_stop()
                    self.action_agent.reset.value = True

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
                print(f"Error in ReplayEnv: {e.with_traceback()}")
                break

        self.action_agent.stop()
        self.stop()
        print("ReplayEnv process stopped")
