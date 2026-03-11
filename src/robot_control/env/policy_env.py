"""Autonomous policy rollout environment — policy network controls the robot."""

import time

import cv2
import numpy as np
import torch

from robot_control.utils.kinematics_utils import (
    trans_mat_to_pos_quat, gripper_raw_to_qpos,
    pos_eef_to_fingertip, pos_fingertip_to_eef,
)
from robot_control.env.xarm_env import XarmEnv

# ---- Episode Constants ----
MAX_TIMESTEP_DEFAULT = 700
CONTROLLER_SWITCH_STEPS = 50

# ---- Image Processing Constants ----
OBS_CROP_SIZE = 400
OBS_RESIZE_SIZE = 200


class PolicyEnv(XarmEnv):
    """Autonomous policy rollout environment.

    Uses a diffusion policy (DPWrapper) to generate actions from camera observations.
    Automatically resets after max_timestep steps.
    """

    def __init__(self, dp_ckpt_path: str, **kwargs):
        super().__init__(action_receiver="policy", **kwargs)

        from lerobot.rrl.dp_wrapper import DPWrapper
        self.policy = DPWrapper(dp_ckpt_path)

    def _read_env_obs(self, rgbs, trans_out, gripper_out):
        """Format observations for policy inference."""
        pos, quat = trans_mat_to_pos_quat(trans_out["value"])
        gripper = gripper_out["value"][0]
        gripper = gripper_raw_to_qpos(gripper)

        obs_state = torch.from_numpy(
            np.concatenate([pos, quat, [gripper]], axis=0)
        ).unsqueeze(0).to("cuda").to(torch.float32)
        obs_state[:, :3] = pos_eef_to_fingertip(obs_state[:, :3], obs_state[:, 3:7])

        img = cv2.cvtColor(rgbs[0], cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        if h < OBS_CROP_SIZE or w < OBS_CROP_SIZE:
            raise ValueError(f"Image too small to crop: {img.shape}")

        cy, cx = h // 2, w // 2
        y0 = cy - OBS_CROP_SIZE // 2
        x0 = cx - OBS_CROP_SIZE // 2
        crop = img[y0:y0 + OBS_CROP_SIZE, x0:x0 + OBS_CROP_SIZE]

        crop_resized = cv2.resize(
            crop, (OBS_RESIZE_SIZE, OBS_RESIZE_SIZE), interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0

        return {
            "observation.state": obs_state,
            "observation.images.front": torch.from_numpy(crop_resized).unsqueeze(0).to("cuda").permute(0, 3, 1, 2).to(torch.float32),
        }

    def run(self) -> None:
        robot_obs_dir, robot_action_dir, rgbs, depths, fps = self._init_run()

        timestep = 0
        max_timestep = MAX_TIMESTEP_DEFAULT

        # Set initial pose
        init_pose = self.init_poses[0].copy()
        self.set_robot_initial_pose(init_pose, fps=fps)
        print("Robot initial pose set")

        reset_done = False
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

                # Auto-reset at max timestep
                if timestep > max_timestep:
                    self.action_agent.reset.value = True

                if self.action_agent.reset.value:
                    timestep = 0
                    init_pose = self.init_poses[0].copy() * np.pi / 180
                    curr_goal = np.array(self.action_agent.command[:].copy())
                    if not reset_done:
                        self.policy.reset()
                        for i in range(CONTROLLER_SWITCH_STEPS):
                            self.action_agent.command[:] = (
                                (i / CONTROLLER_SWITCH_STEPS) * init_pose
                                + (1 - i / CONTROLLER_SWITCH_STEPS) * curr_goal
                            )
                            time.sleep(0.02)
                        reset_done = True
                elif not self.action_agent.reset.value:
                    reset_done = False

                    # Policy step
                    lerobot_obs = self._read_env_obs(rgbs, trans_out, gripper_out)
                    action = self.policy.act(lerobot_obs)
                    actions_eef = action.clone()
                    actions_eef[:, :3] = pos_fingertip_to_eef(
                        actions_eef[:, :3], actions_eef[:, 3:7],
                    )
                    actions_eef = actions_eef.squeeze(0).cpu().numpy()
                    target_qpos = self.get_qpos_from_action_8d(actions_eef, curr_qpos=qpos_out["value"][:7])
                    self.action_agent.command[:] = target_qpos
                    timestep += 1
                    print(f"timestep: {timestep}/{max_timestep}")

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
                print(f"Error in PolicyEnv: {e.with_traceback()}")
                break

        self.action_agent.stop()
        self.stop()
        print("PolicyEnv process stopped")
