"""Residual shared autonomy environment — human teleop + residual policy on top."""

import os
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import yaml
from gymnasium.spaces import Box

from robot_control.utils.kinematics_utils import (
    trans_mat_to_pos_quat, gripper_raw_to_qpos,
    pos_eef_to_fingertip, pos_fingertip_to_eef,
)
from robot_control.utils.math import (
    axis_angle_from_quat, quat_mul, quat_conjugate, quat_from_angle_axis,
    quat_from_matrix,
)
from robot_control.agents.knn_pilot import KNN_Pilot
from robot_control.env.cfg.task_configs import apply_task_offsets
from robot_control.utils.data_storage import store_state_data as _store_state_data
from robot_control.env.xarm_env import XarmEnv

from robot_control.utils.utils import get_root
root: Path = get_root(__file__)

from huggingface_hub import hf_hub_download

HF_DATA_REPO = "shashuo0104/residual_copilot_data"
HF_MODEL_REPO = "shashuo0104/residual_copilot_models"

def _resolve_hf_file(repo_id: str, filename: str, repo_type="dataset", revision=None) -> str:
    return str(Path(hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision,
    )))

# Abbreviated checkpoint_path shortcuts: {Task}_Residual_Copilot or {Task}_Residual_BC
# Maps to HF paths under shared_autonomy_policies/residual_copilot/
_ABBREV_MAP = {
    "Residual_Copilot": "shared_autonomy_policies/residual_copilot/{task}_noisy_knn",
    "Residual_BC": "shared_autonomy_policies/residual_copilot/{task}_bc_teleop",
}
_VALID_TASKS = {"GearMesh", "PegInsert", "NutThread"}

def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    """Resolve abbreviated checkpoint_path like 'GearMesh_Residual_Copilot' to a local HF cache path.

    Returns the original path if it's not an abbreviation.
    """
    parts = checkpoint_path.split("_", 1)
    if len(parts) == 2 and parts[0] in _VALID_TASKS and parts[1] in _ABBREV_MAP:
        task, method = parts
        hf_prefix = _ABBREV_MAP[method].format(task=task)
        # Download the checkpoint and env/agent configs
        for hf_file in [
            f"{hf_prefix}/nn/FactoryXarm.pth",
            f"{hf_prefix}/params/env.yaml",
            f"{hf_prefix}/params/agent.yaml",
        ]:
            local = _resolve_hf_file(HF_MODEL_REPO, hf_file, repo_type="model")
        # Return the directory (3 levels up from the .pth file)
        return str(Path(local).parent.parent)
    return checkpoint_path

def _to_tensor(arr, device='cuda'):
    """Convert numpy array to float32 CUDA tensor with batch dim."""
    return torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=torch.float32)

# ---- Constants (hardware-specific) ----
VELOCITY_SMOOTHING = 0.5
CONTROLLER_SWITCH_STEPS = 50
DEFAULT_DT = 1 / 15  # agent fps
MAX_EPS_LENGTH = 300

class SharedAutonomyEnv(XarmEnv):
    """Residual shared autonomy environment.

    Handles both online (human gello + residual) and offline (KNN_Pilot + residual) modes.
    Requires FoundationPose state estimation for object tracking.
    """

    def __init__(
        self,
        checkpoint_path: str,
        foundation_pose_dir: str,
        offline: bool = False,
        res_bc_path: Union[str, None] = None,
        **kwargs,
    ):
        action_receiver = "residual_offline" if offline else "residual"
        super().__init__(action_receiver=action_receiver, **kwargs)

        self.offline = offline
        checkpoint_path = _resolve_checkpoint_path(checkpoint_path)

        # Load env config from checkpoint
        env_cfg_path = f"{checkpoint_path}/params/env.yaml"
        if os.path.exists(env_cfg_path):
            with open(env_cfg_path, "r") as f:
                env_cfg = yaml.full_load(f)
            ctrl = env_cfg.get("ctrl", {})
        else:
            print(f"[Warning] No env.yaml at {env_cfg_path}, using defaults")
            env_cfg = {}
            ctrl = {}

        self.residual_pos_scale = ctrl.get("res_pos_action_threshold", [0.03])[0]
        self.residual_rot_scale = ctrl.get("res_rot_action_threshold", [0.5])[0]
        self.residual_grip_scale = ctrl.get("res_gripper_action_threshold", [0.1])[0]
        self.residual_smoothing = ctrl.get("ema_factor", 0.2)

        # Obs/action dims from env cfg (residual obs = residual_obs_order dims + prev_actions)
        residual_obs_order = env_cfg.get("residual_obs_order", [])
        _OBS_DIM_MAP = {
            "fingertip_pos": 3, "fingertip_quat": 4, "gripper": 1,
            "fingertip_pos_rel_fixed": 3, "fingertip_pos_rel_held": 3,
            "ee_linvel": 3, "ee_angvel": 3,
            "base_fingertip_pos": 3, "base_fingertip_quat": 4, "base_gripper": 1,
        }
        residual_action_dim = env_cfg.get("residual_action_space", 7)
        if residual_obs_order:
            obs_dim = sum(_OBS_DIM_MAP[k] for k in residual_obs_order) + residual_action_dim
        else:
            obs_dim = 35  # fallback: 28 obs + 7 prev_actions
        self._obs_dim = obs_dim
        self._action_dim = residual_action_dim

        # State estimator
        from robot_control.perception.state_estimator import StateEstimator
        self.state_estimator = StateEstimator(foundation_pose_dir=foundation_pose_dir)

        # Residual copilot policy
        self.residual_copilot = self._load_actor_mlp(
            agent_yaml=f"{checkpoint_path}/params/agent.yaml",
            checkpoint_path=f"{checkpoint_path}/nn/FactoryXarm.pth",
        )

        # Optional residual BC policy
        if res_bc_path is not None:
            res_bc_path = _resolve_checkpoint_path(res_bc_path)
            self.residual_bc = self._load_actor_mlp(
                agent_yaml=f"{res_bc_path}/params/agent.yaml",
                checkpoint_path=f"{res_bc_path}/nn/FactoryXarm.pth",
            )

        # KNN pilot (base action source for offline mode)
        knn_cfg_path = str(Path(__file__).parent / "cfg" / "knn_pilot_default.json")
        task_cfg = env_cfg.get("task", {})
        base_rand = env_cfg.get("base_rand", {})
        hf_repo = task_cfg.get("hf_repo", HF_DATA_REPO)
        hf_file = task_cfg.get("train_data_hf_file", "teleop/peginsert_train_data.npy")
        knn_override = {}
        horizon = base_rand.get("horizon")
        if horizon and len(horizon) == 2:
            knn_override["min_horizon"] = horizon[0]
            knn_override["max_horizon"] = horizon[1]
        self.base_actions_agent = KNN_Pilot(
            cfg_path=knn_cfg_path,
            data_path=_resolve_hf_file(hf_repo, hf_file),
            num_envs=1,
            device="cuda",
            cfg_override=knn_override or None,
        )

        # Task name from foundation_pose_dir
        self.task_name = os.path.basename(foundation_pose_dir)
        assert self.task_name in ["gearmesh", "peginsert", "nutthread"], \
            "Task name must be one of ['gearmesh', 'peginsert', 'nutthread']"

    def _load_actor_mlp(self, agent_yaml: str, checkpoint_path: str):
        """Load residual policy model from YAML config and checkpoint."""
        from rl_games.algos_torch.players import PpoPlayerContinuous

        with open(agent_yaml, "r") as f:
            agent_cfg = yaml.safe_load(f)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["resume_path"] = checkpoint_path
        agent_cfg["params"]["config"]["env_info"] = {
            "observation_space": Box(-np.inf, np.inf, (self._obs_dim,), dtype=np.float32),
            "action_space": Box(-1.0, 1.0, (self._action_dim,), dtype=np.float32),
            "agents": 1,
            "value_size": 1,
        }

        policy = PpoPlayerContinuous(agent_cfg["params"])
        policy.restore(checkpoint_path)
        policy.reset()
        return policy

    def _get_base_actions(self, gello_actions, fingertip_pos):
        """Compute base actions from gello/knn actions, updating position delta tracking."""
        base_actions = gello_actions.clone()
        last_gello = gello_actions[:, :3].clone()
        if self.last_gello_pos is not None:
            base_actions[:, :3] = (gello_actions[:, :3] - self.last_gello_pos) + fingertip_pos
        self.last_gello_pos = last_gello
        return base_actions

    def _compute_velocity(self, fingertip_pos, fingertip_quat, dt, device='cuda'):
        """Finite-difference EE velocity with exponential smoothing."""
        if self.prev_fingertip_pos is None:
            linvel = torch.zeros((1, 3), dtype=torch.float32, device=device)
            angvel = torch.zeros((1, 3), dtype=torch.float32, device=device)
        else:
            linvel = (fingertip_pos - self.prev_fingertip_pos) / dt
            rot_diff = quat_mul(fingertip_quat, quat_conjugate(self.prev_fingertip_quat))
            rot_diff *= torch.sign(rot_diff[:, 0]).unsqueeze(-1)
            angvel = axis_angle_from_quat(rot_diff) / dt
            a = VELOCITY_SMOOTHING
            linvel = a * self.last_linvel + (1 - a) * linvel
            angvel = a * self.last_angvel + (1 - a) * angvel

        self.last_linvel = linvel.clone()
        self.last_angvel = angvel.clone()
        self.prev_fingertip_pos = fingertip_pos
        self.prev_fingertip_quat = fingertip_quat
        return linvel, angvel

    def get_residual_observations(self, trans_out, gripper_out, dt,
                                  held_pos, held_mat, fixed_pos, fixed_mat,
                                  action_trans_out, action_gripper_out, action_qpos_out,
                                  device='cuda'):
        """Compute residual observation vector from robot state and object poses."""
        capture_time = trans_out['capture_time']

        # Robot state
        curr_pos, curr_quat = trans_mat_to_pos_quat(trans_out["value"])
        fingertip_quat = _to_tensor(curr_quat, device)
        fingertip_pos = pos_eef_to_fingertip(_to_tensor(curr_pos, device), fingertip_quat, device=device)
        gripper = torch.tensor([[gripper_raw_to_qpos(gripper_out["value"][0])]], dtype=torch.float32, device=device)

        # Object poses
        fixed_quat = quat_from_matrix(_to_tensor(fixed_mat, device))
        held_quat = quat_from_matrix(_to_tensor(held_mat, device))
        fixed_pos, held_pos = apply_task_offsets(
            self.task_name, _to_tensor(fixed_pos, device), fixed_quat,
            _to_tensor(held_pos, device), held_quat, device=device,
        )

        # Velocity
        linvel, angvel = self._compute_velocity(fingertip_pos, fingertip_quat, dt, device)

        # Base actions (from KNN pilot or gello teleop)
        if self.offline:
            eps_idx = torch.tensor([0], dtype=torch.int64, device=device)
            gello_actions = self.base_actions_agent.get_actions(eps_idx, fingertip_pos, fingertip_quat, gripper)
        else:
            gello_pos, gello_quat = trans_mat_to_pos_quat(action_trans_out["value"])
            gello_pos = _to_tensor(gello_pos, device)
            gello_quat = _to_tensor(gello_quat, device)
            gello_gripper = torch.tensor([[action_gripper_out["value"][0]]], dtype=torch.float32, device=device)
            gello_actions = torch.cat([gello_pos, gello_quat, gello_gripper], dim=-1)
            gello_actions[:, :3] = pos_eef_to_fingertip(gello_actions[:, :3], gello_actions[:, 3:7], device=device)

        base_actions = self._get_base_actions(gello_actions, fingertip_pos)

        obs = torch.cat([
            fingertip_pos, fingertip_quat, gripper,
            fingertip_pos - fixed_pos, fingertip_pos - held_pos,
            linvel, angvel,
            gello_actions, self.prev_actions,
        ], dim=-1).to(torch.float32)

        return capture_time, obs, fixed_pos, held_pos, base_actions[:, :3]

    def apply_residual(self, residual, base_actions, qpos, device='cuda'):
        """Apply residual actions to base actions and convert to joint space."""
        pos_actions = residual[:, 0:3] * self.residual_pos_scale
        rot_actions = residual[:, 3:6] * self.residual_rot_scale
        grip_actions = residual[:, 6:7] * self.residual_grip_scale

        target_pos = base_actions[:, 0:3] + pos_actions

        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_quat = quat_from_angle_axis(angle, axis)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device).unsqueeze(0)
        rot_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > 1e-6, rot_quat, identity)
        target_quat = quat_mul(rot_quat, base_actions[:, 3:7])
        target_gripper = torch.clamp(base_actions[:, 7:8] + grip_actions, 0.0, 1.0)

        actions_fingertip = torch.cat([target_pos, target_quat, target_gripper], dim=-1)
        goal_pos = actions_fingertip[:, :3].clone()

        actions_eef = actions_fingertip.clone()
        actions_eef[:, :3] = pos_fingertip_to_eef(actions_eef[:, :3], actions_eef[:, 3:7], device=device)
        target_qpos = self.get_qpos_from_action_8d(actions_eef.squeeze(0).cpu().numpy(), curr_qpos=qpos[:7])

        return actions_fingertip, target_qpos, goal_pos

    def _run_residual_step(self, state, held_pos, held_mat, fixed_pos, fixed_mat,
                           residual, use_residual, get_obs_time):
        """Execute one residual control step. Returns updated loop state."""
        if get_obs_time is None:
            dt = DEFAULT_DT
        else:
            dt = time.time() - get_obs_time
        get_obs_time = time.time()

        self.prev_actions = residual
        capture_time, obs, fixed_pos_obs, held_pos_obs, base_pos = \
            self.get_residual_observations(
                state["trans_out"], state["gripper_out"], dt,
                held_pos, held_mat, fixed_pos, fixed_mat,
                state["action_trans_out"], state["action_gripper_out"], state["action_qpos_out"],
            )

        fingertip_pos = obs[0, 0:3].cpu().numpy()

        # Select policy
        if self.action_agent.use_residual_copilot.value:
            residual = self.residual_copilot.get_action(obs, is_deterministic=True)
        elif self.action_agent.use_residual_bc.value:
            assert hasattr(self, 'residual_bc'), "Residual BC model not loaded!"
            residual = self.residual_bc.get_action(obs, is_deterministic=True)
        residual = residual * self.residual_smoothing + self.prev_actions * (1 - self.residual_smoothing)

        # Handle controller switching (teleop <-> residual)
        if use_residual == self.action_agent.use_teleop.value:
            self._switch_controller()

        use_residual = not self.action_agent.use_teleop.value
        actions_fingertip, actions_qpos, final_pos = self.apply_residual(
            residual * use_residual, obs[:, -15:-7], state["qpos_out"]["value"],
        )

        return {
            "residual": residual,
            "use_residual": use_residual,
            "actions_fingertip": actions_fingertip,
            "actions_qpos": actions_qpos,
            "capture_time": capture_time,
            "obs": obs,
            "fingertip_pos": fingertip_pos,
            "base_pos": base_pos.cpu().numpy(),
            "final_pos": final_pos[0].cpu().numpy(),
            "fixed_pos_obs": fixed_pos_obs,
            "held_pos_obs": held_pos_obs,
            "get_obs_time": get_obs_time,
        }

    def _switch_controller(self):
        """Smoothly interpolate between teleop and residual commands."""
        print("switching controller...")
        res_comm = np.array(self.action_agent.command_with_residual[:].copy())
        teleop_comm = np.array(self.action_agent.command[:].copy())
        self.action_agent.switching_controller.value = True
        for i in range(1, CONTROLLER_SWITCH_STEPS):
            t = i / CONTROLLER_SWITCH_STEPS
            if not self.action_agent.use_teleop.value:
                self.action_agent.command_with_residual[:] = t * res_comm + (1 - t) * teleop_comm
            else:
                self.action_agent.command[:] = t * teleop_comm + (1 - t) * res_comm
            time.sleep(1 / 50)
        self.action_agent.switching_controller.value = False

    def _annotate_residual_overlays(self, img, fingertip_pos, base_pos, final_pos,
                                    held_pos_obs, fixed_pos_obs):
        """Draw residual control overlays on camera image."""
        try:
            if not self.action_agent.use_teleop.value:
                img = self.state_estimator.draw_triangle_from_base_points(img, fingertip_pos, base_pos, final_pos)
            if self.last_gello_pos is not None:
                img = self.state_estimator.project_points_on_image(img, self.last_gello_pos.cpu().numpy())
            img = self.state_estimator.project_points_on_image(img, held_pos_obs.cpu().numpy(), color=(255, 0, 0))
            img = self.state_estimator.project_points_on_image(img, fixed_pos_obs.cpu().numpy(), color=(0, 255, 0))
            img = self.state_estimator.draw_mask_regions(img)
        except BaseException as e:
            print(f"Error in drawing overlays: {e.with_traceback()}")

        if self.action_agent.use_residual_copilot.value:
            text = "Residual Copilot"
        elif self.action_agent.use_residual_bc.value:
            text = "Residual BC"
        else:
            text = "Teleop"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

    def _get_mode(self):
        if self.action_agent.use_residual_copilot.value:
            return "residual_copilot"
        elif self.action_agent.use_residual_bc.value:
            return "residual_bc"
        return "teleop"

    def _reset_loop_state(self):
        """Reset per-episode loop state for residual control."""
        self.prev_fingertip_pos = None
        self.prev_fingertip_quat = None
        self.last_gello_pos = None
        self.last_linvel = None
        self.last_angvel = None
        self.base_actions_agent.clear([0])

    def run(self) -> None:
        robot_obs_dir, robot_action_dir, rgbs, depths, fps = self._init_run()

        timestep = 0
        episode = 0
        self._reset_loop_state()
        get_obs_time = None
        residual = torch.zeros((1, 7), dtype=torch.float32, device='cuda')
        use_residual = self.action_agent.use_residual_copilot.value or self.action_agent.use_residual_bc.value
        fixed2cam = None

        # Offline state: "resetting" -> "paused" -> "running" -> "resetting" ...
        init_pose = (self.init_poses[0].copy() * np.pi / 180).tolist()
        reset_step = 0  # >0 means interpolating to init pose
        reset_start_command = None
        paused = self.offline  # offline starts paused; online always runs

        if self.offline:
            # Blocking only for the very first time (robot not at init pose yet)
            self.set_robot_initial_pose(self.init_poses[0].copy(), fps=fps)
            print("Robot at initial pose. Press 1/2/3 to select agent, 'm' to retrack, then 's' to start.")

        time.sleep(1)

        while self.alive:
            try:
                tic = time.time()
                state = self._read_state()
                self._handle_recording_signals(mode=self._get_mode())

                perception_out = state.get("perception_out")
                self._update_images(perception_out, rgbs, depths)

                # Object pose estimation — always runs
                if perception_out is not None and self.state_estimator is not None:
                    retrack = self.action_agent.track_obj.value
                    held2cam = self.state_estimator.estimate_object_poses(
                        rgbs[0], depths[0], retrack=retrack, obj_name="held_asset")
                    if fixed2cam is None or retrack:
                        fixed2cam = self.state_estimator.estimate_object_poses(
                            rgbs[0], depths[0], retrack=retrack, obj_name="fixed_asset")
                    if retrack:
                        self.action_agent.track_obj.value = False

                    held2base = self.state_estimator.cam2base @ held2cam
                    fixed2base = self.state_estimator.cam2base @ fixed2cam
                    held_pos, held_mat = held2base[:3, 3], held2base[:3, :3]
                    fixed_pos, fixed_mat = fixed2base[:3, 3], fixed2base[:3, :3]

                # Residual inference — always runs (obs, policy, overlays)
                step = self._run_residual_step(
                    state, held_pos, held_mat, fixed_pos, fixed_mat,
                    residual, use_residual, get_obs_time,
                )
                residual = step["residual"]
                use_residual = step["use_residual"]
                get_obs_time = step["get_obs_time"]

                # Offline: state machine for what action to send
                if self.offline:
                    # Auto-reset at max episode length
                    if not paused and reset_step == 0 and timestep >= MAX_EPS_LENGTH:
                        self.action_agent.reset.value = True

                    # 'r' pressed — begin interpolation to init pose
                    if self.action_agent.reset.value:
                        self.action_agent.reset.value = False
                        if reset_step == 0 and not paused:
                            self.perception.set_record_stop()
                            self._reset_loop_state()
                            timestep = 0
                            residual = torch.zeros((1, 7), dtype=torch.float32, device='cuda')
                            episode += 1
                            reset_start_command = list(self.action_agent.command[:])
                            reset_step = 1
                            print(f"Returning to initial pose...")

                    # Interpolating to init pose (non-blocking, one step per loop)
                    if reset_step > 0:
                        t = reset_step / CONTROLLER_SWITCH_STEPS
                        interp = [t * ip + (1 - t) * sc for ip, sc in zip(init_pose, reset_start_command)]
                        self.action_agent.command[:] = interp
                        reset_step += 1
                        if reset_step > CONTROLLER_SWITCH_STEPS:
                            reset_step = 0
                            paused = True
                            print(f"Reset. Press 1/2/3 to select agent, 'm' to retrack, then 's' to start.")

                    elif paused:
                        # Hold init pose with zero residual; 's' starts episode
                        residual = torch.zeros((1, 7), dtype=torch.float32, device='cuda')
                        self.action_agent.command[:] = init_pose
                        if self.action_agent.record_start.value:
                            self.action_agent.record_start.value = False
                            paused = False
                            print(f"Starting episode {episode}")

                    else:
                        # Running — send policy actions
                        self.action_agent.command[:] = step["actions_qpos"]
                        timestep += 1
                        print(f"episode {episode} timestep: {timestep}/{MAX_EPS_LENGTH}")

                        if state.get("trans_out") is not None and state.get("action_qpos_out") is not None:
                            _store_state_data(
                                step["obs"], state["qpos_out"]["value"],
                                step["actions_fingertip"], step["actions_qpos"],
                                step["capture_time"], robot_obs_dir, robot_action_dir,
                            )
                else:
                    # Online mode — always send policy actions
                    self.action_agent.command_with_residual[:] = step["actions_qpos"]

                    if state.get("trans_out") is not None and state.get("action_qpos_out") is not None:
                        _store_state_data(
                            step["obs"], state["qpos_out"]["value"],
                            step["actions_fingertip"], step["actions_qpos"],
                            step["capture_time"], robot_obs_dir, robot_action_dir,
                        )

                # Overlays + display — always runs
                rgbs[0] = self._annotate_residual_overlays(
                    rgbs[0], step["fingertip_pos"], step["base_pos"], step["final_pos"],
                    step["held_pos_obs"], step["fixed_pos_obs"],
                )

                mode = self._get_mode()
                text = {"residual_copilot": "Residual Copilot", "residual_bc": "Residual BC"}.get(mode, "Teleop")
                if self.offline and (paused or reset_step > 0):
                    text += " (resetting)" if reset_step > 0 else " (paused)"
                cv2.putText(rgbs[0], text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self._build_display_frame(rgbs, depths)

                time.sleep(max(0, 1 / fps - (time.time() - tic)))
                actual_fps = 1 / (time.time() - tic)
                if abs(actual_fps - fps) > 1:
                    print(f"real env fps: {actual_fps:.1f} (target: {fps})")

            except BaseException as e:
                print(f"Error in SharedAutonomyEnv: {e.with_traceback()}")
                break

        self.action_agent.stop()
        self.stop()
        print("SharedAutonomyEnv process stopped")
