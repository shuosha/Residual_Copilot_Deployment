import threading
import multiprocess as mp
import time
from enum import Enum
import traceback

import numpy as np
import transforms3d
import copy
import os, sys, contextlib

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
    from xarm.wrapper import XArmAPI

from robot_control.utils.kinematics_utils import KinHelper
from robot_control.utils.udp_util import udpReceiver, udpSender
from robot_control.control.common.communication import XARM_STATE_PORT, XARM_CONTROL_PORT, XARM_CONTROL_PORT_L, XARM_CONTROL_PORT_R
from robot_control.control.common.xarm import *

np.set_printoptions(precision=2, suppress=True)

class Rate:
    """Rate limiter for controlling loop execution frequency."""
    
    def __init__(self, *, duration):
        """Initialize rate limiter with target duration between sleeps."""
        self.duration = duration
        self.last = time.time()

    def sleep(self, duration=None) -> None:
        """Sleep to maintain target rate, adjusting for elapsed time."""
        duration = self.duration if duration is None else duration
        assert duration >= 0
        now = time.time()
        passed = now - self.last
        remaining = duration - passed
        assert passed >= 0
        if remaining > 0.0001:
            time.sleep(remaining)
        self.last = time.time()


class ControllerState(Enum):
    INIT = 0
    RUNNING = 1
    STOP = 2

class XarmController(mp.Process):
    
    name = "xarm_controller"
    teleop_kin_helper = KinHelper(robot_name="xarm7")
    robot_kin_helper = KinHelper(robot_name="xarm7")

    XY_MIN, XY_MAX = XY_MIN, XY_MAX 
    X_MIN, X_MAX = X_MIN, X_MAX 
    Y_MIN, Y_MAX = Y_MIN, Y_MAX 
    Z_MIN, Z_MAX = Z_MIN, Z_MAX 
    
    MOVE_RATE = MOVE_RATE 
    MOVE_SLEEP = MOVE_SLEEP 
    XYZ_VELOCITY = XYZ_VELOCITY 
    ANGLE_VELOCITY_MAX = ANGLE_VELOCITY_MAX 

    GRIPPER_OPEN_MAX = GRIPPER_OPEN_MAX
    GRIPPER_OPEN_MIN = GRIPPER_OPEN_MIN

    POSITION_UPDATE_FREQ = POSITION_UPDATE_FREQ 
    COMMAND_EXECUTION_FREQ = COMMAND_EXECUTION_FREQ 
    
    VELOCITY_DELTA_SCALE = VELOCITY_DELTA_SCALE
    MAX_ACTIVATE_DELTA = MAX_ACTIVATE_DELTA
    MAX_DELTA_NORM = MAX_DELTA_NORM


    def log(self, msg):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            self.pprint(msg)

    @staticmethod
    def wrap_to_pi(x):
        """Map angles to [-pi, pi). Works elementwise for numpy arrays."""
        return (x + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def interpolate(curr, nxt, alpha):
        """
        curr: (7,) current joint angles [rad]
        nxt:  (7,) target joint angles [rad]
        alpha: scalar in [0, 1]
        returns: (7,) interpolated joint angles along shortest arc
        """
        curr = np.asarray(curr, dtype=np.float64)
        nxt  = np.asarray(nxt,  dtype=np.float64)
        alpha = float(alpha)

        delta = XarmController.wrap_to_pi(nxt - curr)          # shortest signed delta per joint
        out = curr + alpha * delta
        return XarmController.wrap_to_pi(out)                  # optional normalization

    @staticmethod
    def pprint(*args, **kwargs):
        """Print formatted message with timestamp and caller information."""
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print(
                "[{}][{}] {}".format(
                    time.strftime("\033[92m%Y-%m-%d %H:%M:%S\033[0m", time.localtime(time.time())),
                    stack_tuple[1],
                    " ".join(map(str, args)),
                )
            )
        except:
            print(*args, **kwargs)

    def __init__(
        self,
        start_time,
        init_pose=[256.7, 5.1, 400.1, 178.9, 0.0, 1.4],
        init_servo_angle=[0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0],
        ip="192.168.1.196", 
        robot_id=0,
        control_mode="position_control",  # "position_control" or "velocity_control"
        robot_name="xarm7", # "xarm7" or "uf850"
        admittance_control=False,
        ema_factor=1.0,
        gripper_enable=False,
        comm_update_fps=30.0,
        verbose=False,
    ):
        """Initialize XArm controller process with specified control mode, robot configuration, and communication settings."""
        self.robot_id = robot_id
        self.init_pose = init_pose
        self.init_servo_angle = init_servo_angle
        self.control_mode = control_mode
        assert control_mode in ["position_control", "velocity_control", "cartesian_position_planning"], "control_mode must be position_control, velocity_control or cartesian_position_planning"
        self.admittance_control = admittance_control
        if admittance_control:
            assert control_mode != "velocity_control", "admittance control is not compatible with velocity control mode"
        self.ema_factor = ema_factor
        self.comm_update_fps = comm_update_fps

        assert robot_name in ["xarm7", "uf850"], "robot_name must be xarm7 or uf850"

        # assert mode == "2D", "currently only support 2D mode"
        super().__init__()
        
        self.start_time = start_time
        self._ip = ip
        self.gripper_enable = gripper_enable
        # input(f"Connecting to {self._ip}, Press Enter to continue...")
        self.verbose = verbose
        self.exe_lock = mp.Lock()

        self.state = mp.Value('i', ControllerState.INIT.value)

        # self.command_q = mp.Queue()
        self.cur_trans_q = mp.Array('d', [0.0] * 16)
        self.cur_qpos_q = mp.Array('d', [0.0] * 7)
        self.cur_gripper_q = mp.Array('d', [0.0] * 1)
        self.cur_force_q = mp.Array('d', [0.0] * 6)
        self.cur_time_q = mp.Value('d', 0.0)

        self.command_receiver = None
        # self.state_sender = None

        self.cur_gripper_pos = None
        self.cur_qpos = None

        self.teleop_activated = mp.Value('b', False)  # whether the teleop is activated, used to control the gripper

    # ======= xarm controller queue START =======
    def update_cur_position(self):
        """Update current arm position in a separate thread using forward kinematics for accurate position tracking."""

        update_rate = Rate(
            duration=1/self.POSITION_UPDATE_FREQ,
        )

        if self.robot_id == -1:
            self.state_sender = udpSender(port=XARM_STATE_PORT)
        try:
            while self.state.value in [ControllerState.INIT.value, ControllerState.RUNNING.value]:
                update_start_time = time.time()

                if self.gripper_enable:
                    cur_gripper_pos = self.get_gripper_state()
                    self.cur_gripper_q[:] = np.array([cur_gripper_pos])
                    self.cur_gripper_pos = cur_gripper_pos

                cur_qpos = np.array(self._arm.get_servo_angle()[1][0:7]) / 180. * np.pi
                
                current_time = time.time()
                self.cur_time_q.value = current_time

                fk_trans_mat = self.robot_kin_helper.compute_fk_sapien_links(cur_qpos, [self.robot_kin_helper.sapien_eef_idx])[0]
                cur_xyzrpy = np.zeros(6)
                cur_xyzrpy[:3] = fk_trans_mat[:3, 3] * 1000
                cur_xyzrpy[3:] = transforms3d.euler.mat2euler(fk_trans_mat[:3, :3])
                cur_xyzrpy[3:] = cur_xyzrpy[3:] / np.pi * 180
                
                # self.log(f"d_pos: {self.arm.get_position()[1][0:6]-cur_position}")

                # always update the latest position
                self.cur_trans_q[:] = fk_trans_mat.flatten()
                self.cur_qpos_q[:] = cur_qpos

                code, ft_data = self._arm.get_ft_sensor_data()
                if code == 0:
                    self.cur_force_q[:] = np.array(ft_data)
                else:
                    raise ValueError("get_ft_sensor_data Error", code, ft_data)

                self.cur_qpos = cur_qpos

                if self.robot_id == -1:
                    state = {
                        # "e2b": fk_trans_mat,
                        "pos": cur_xyzrpy,
                        "qpos": cur_qpos,
                    }
                    if self.gripper_enable:
                        state["gripper"] = cur_gripper_pos
                    self.state_sender.send(state)

                update_rate.sleep()

        except Exception as e:
            self.pprint(f"update_cur_position error")
            self.state.value = ControllerState.STOP.value
        finally:
            # self.state_sender.close()
            # self.command_q.close()
            # self.cur_trans_q.close()
            # self.cur_qpos_q.close()
            # self.cur_gripper_q.close()
            print("update_cur_position exit!")

    # NOTE: The following methods are not used externally but kept for potential future use
    def get_current_joint(self):
        """Get current joint positions in radians (deep copy). Not used externally."""
        return copy.deepcopy(self.cur_qpos)
    
    def get_current_gripper(self):
        """Get current gripper position (deep copy). Not used externally."""
        return copy.deepcopy(self.cur_gripper_pos)

    def get_current_joint_deg(self):
        """Get current joint positions in degrees (deep copy). Not used externally."""
        return copy.deepcopy(self.cur_qpos) / np.pi * 180

    def get_current_pose(self):
        """Get current end-effector pose (not implemented). Not used externally."""
        raise NotImplementedError
        return self.cur_xyzrpy

    # ======= xarm controller queue END =======


    # ======= xarm SDK API wrapper START =======
    def open_gripper(self, wait=True):
        """Open gripper to maximum openness."""
        return self.set_gripper_openness(self.GRIPPER_OPEN_MAX, wait=wait)

    def close_gripper(self, wait=True):
        """Close gripper to minimum openness."""
        return self.set_gripper_openness(self.GRIPPER_OPEN_MIN, wait=wait)

    def set_gripper_openness(self, openness, wait=True):
        """Set gripper openness level and optionally wait for completion."""
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code = self._arm.set_gripper_position(openness, wait=wait)
        if not self._check_code(code, "set_gripper_position"):
            raise ValueError("close_gripper Error")
        return True

    def get_gripper_state(self):
        """Get current gripper position state."""
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code, state = self._arm.get_gripper_position()
        if not self._check_code(code, "get_gripper_position"):
            raise ValueError("get_gripper_position Error")
        return state

    def _error_warn_changed_callback(self, data):
        """Callback for error/warning changes: sets alive to False on error."""
        if data and data["error_code"] != 0:
            self.alive = False
            self.pprint("err={}, quit".format(data["error_code"]))
            self._arm.release_error_warn_changed_callback(
                self._error_warn_changed_callback
            )

    def _state_changed_callback(self, data):
        """Callback for state changes: sets alive to False when state is 4 (emergency stop)."""
        if data and data["state"] == 4:
            self.alive = False
            self.pprint("state=4, quit")
            self._arm.release_state_changed_callback(self._state_changed_callback)
    
    def _count_changed_callback(self, data):
        """Callback for counter value changes: logs counter value if robot is alive."""
        if self.is_alive:
            self.pprint("counter val: {}".format(data["count"]))

    def _check_code(self, code, label):
        """Check XArm API return code and handle errors by setting alive to False."""
        if not self.is_alive or code != 0:
            # import ipdb; ipdb.set_trace()
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint(
                "{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}".format(
                    label,
                    code,
                    self._arm.connected,
                    self._arm.state,
                    self._arm.error_code,
                    ret1,
                    ret2,
                )
            )
        return self.is_alive
    # ======= xarm SDK API wrapper END =======

    # ======= xarm control START =======
    def _init_robot(self):
        """Initialize robot: set control mode, enable gripper if needed, and configure admittance control if enabled."""
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        if self.control_mode == "position_control":
            mode = 1 # streaming position control == servo mode
        elif self.control_mode == "velocity_control":
            mode = 4 # streaming velocity control
        elif self.control_mode == "cartesian_position_planning":
            raise NotImplementedError("cartesian control is no longer used")
        else:
            raise ValueError("Invalid control mode")
        self._arm.set_mode(mode)  # NOTE: 0: position control mode, 1: servo control mode, 4: velocity control mode
        self._arm.set_state(0)
        if self.gripper_enable:
            self._arm.set_gripper_mode(0)
            self._arm.set_gripper_enable(True)
            self._arm.clean_gripper_error()
        # self._arm.set_collision_sensitivity(1)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(
            self._error_warn_changed_callback
        )
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, "register_count_changed_callback"):
            self._arm.register_count_changed_callback(self._count_changed_callback)

        if self.admittance_control:
            assert self.control_mode != "velocity_control", "admittance control is not compatible with velocity control mode"
            self._arm.clean_error()
            self._arm.clean_warn()

            # set tool admittance parameters:
            K_pos = 1000       #  x/y/z linear stiffness coefficient, range: 0 ~ 2000 (N/m)
            K_ori = 10           #  Rx/Ry/Rz rotational stiffness coefficient, range: 0 ~ 20 (Nm/rad) 

            # Attention: for M and J, smaller value means less effort to drive the arm, but may also be less stable, please be careful. 
            M = float(0.1)         #  x/y/z equivalent mass; range: 0.02 ~ 1 kg
            J = float(1e-3)         #  Rx/Ry/Rz equivalent moment of inertia, range: 1e-4 ~ 0.01 (Kg*m^2)

            c_axis = [1,1,1,1,1,1] # set z axis as compliant axis
            ref_frame = 1         # 0 : base , 1 : tool

            self._arm.set_ft_sensor_admittance_parameters(coord=ref_frame, c_axis=c_axis, M=[M, M, M, J, J, J], K=[K_pos, K_pos, K_pos, K_ori, K_ori, K_ori], B=[0]*6) # B(damping) is reserved, give zeros

            # enable ft sensor communication
            self._arm.set_ft_sensor_enable(1)
            # will overwrite previous sensor zero and payload configuration
            self._arm.set_ft_sensor_zero() # remove this if zero_offset and payload already identified & compensated!
            time.sleep(0.2) # wait for writing zero operation to take effect, do not remove #TODO: test longer time

            # move robot in admittance control application
            self._arm.set_ft_sensor_mode(1)
            self._arm.set_ft_collision_rebound(0)
            # will start after set_state(0)
            self._arm.set_state(0)

            code, config = self._arm.get_ft_sensor_config()
            # if code == 0:
            #     print('ft_mode: {}'.format(config[0]))
            #     print('ft_is_started: {}'.format(config[1]))
            #     print('ft_type: {}'.format(config[2]))
            #     print('ft_id: {}'.format(config[3]))
            #     print('ft_freq: {}'.format(config[4]))
            #     print('ft_mass: {}'.format(config[5]))
            #     print('ft_dir_bias: {}'.format(config[6]))
            #     print('ft_centroid: {}'.format(config[7]))
            #     print('ft_zero: {}'.format(config[8]))
            #     print('imp_coord: {}'.format(config[9]))
            #     print('imp_c_axis: {}'.format(config[10]))
            #     print('M: {}'.format(config[11]))
            #     print('K: {}'.format(config[12]))
            #     print('B: {}'.format(config[13]))
            #     print('f_coord: {}'.format(config[14]))
            #     print('f_c_axis: {}'.format(config[15]))
            #     print('f_ref: {}'.format(config[16]))
            #     print('f_limits: {}'.format(config[17]))
            #     print('kp: {}'.format(config[18]))
            #     print('ki: {}'.format(config[19]))
            #     print('kd: {}'.format(config[20]))
            #     print('xe_limit: {}'.format(config[21]))

        
        self.state.value = ControllerState.RUNNING.value

    def preprocess_command(self, arm_state_goal, curr_arm_state):
        """Preprocess command: check teleop activation and apply safety limits on joint deltas."""
        # check teleop activated
        delta = arm_state_goal - curr_arm_state # joint space + gripper
        joint_delta_norm = np.linalg.norm(delta)
        max_joint_delta = np.abs(delta).max()

        if not self.teleop_activated.value:
            if max_joint_delta < self.MAX_ACTIVATE_DELTA:
                self.teleop_activated.value = True
            next_arm_state = curr_arm_state
        else:
            if joint_delta_norm > self.MAX_DELTA_NORM:
                delta = delta / joint_delta_norm * self.MAX_DELTA_NORM # upper bounds delta at: max_delta_norm
                # print("- - - action smoothed - - -")
            next_arm_state = curr_arm_state + delta

        return next_arm_state


    # TODO: add reset method that main process can call

    def velocity_control(self, next_state, current_state, ema_factor=1.0, ignore_error=False):
        """Execute velocity control by sending streaming velocity targets to robot."""
        # NOTE: velocity control don't use ema
        # next_joints = ema_factor * next_joints + (1 - ema_factor) * current_joints

        # NOTE: delta for velocity control
        next_joints = next_state[0:7] - current_state[0:7]
        
        # denormalize gripper position
        if self.gripper_enable:
            gripper_pos = next_state[-1]
            denormalized_gripper_pos = gripper_pos * (GRIPPER_OPEN_MIN - GRIPPER_OPEN_MAX) + GRIPPER_OPEN_MAX

        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        if self.gripper_enable and len(next_joints) == 7: #TODO: hardcoded right now
            isclose = np.isclose(self.cur_gripper_pos, denormalized_gripper_pos)
            if not isclose:
                self.set_gripper_openness(denormalized_gripper_pos, wait=False)

        v = next_joints * self.VELOCITY_CONTROL_SCALE
        v = v.tolist()
        # print("set velocity:", np.round(v,4), "shape:", np.array(v).shape)
        code = self._arm.vc_set_joint_velocity(v, is_radian=True, is_sync=False, duration=0)

        if not self._check_code(code, "vc_set_joint_velocity"):
            raise ValueError("velocity control error")
        if ignore_error:
            self._arm.clean_error()
            self._arm.clean_warn()

    def position_control(self, next_arm_goal, prev_arm_goal, next_gripper=None, ema_factor=0.5, ignore_error=True):
        """Execute position control by sending streaming position/servo targets with EMA smoothing."""
        next_arm_goal = ema_factor * next_arm_goal + (1 - ema_factor) * prev_arm_goal

        # denormalize gripper position
        if self.gripper_enable:
            assert next_gripper is not None, "next_gripper must be provided when gripper_enable is True"
            denormalized_gripper_pos = next_gripper * (GRIPPER_OPEN_MIN - GRIPPER_OPEN_MAX) + GRIPPER_OPEN_MAX

        if not self.is_alive:
            print("next_arm_goal:", next_arm_goal)
            print("prev_arm_goal:", prev_arm_goal)
            if self.gripper_enable:
                print("next_gripper:", next_gripper)

            raise ValueError("Robot is not alive!")
    
        if self.gripper_enable and len(next_arm_goal) == 7 and self.cur_gripper_pos is not None: #TODO: hardcoded right now            
            if np.abs(self.cur_gripper_pos - denormalized_gripper_pos) > 5.0:
                self._arm.set_gripper_position(denormalized_gripper_pos, wait=False, speed=2500)

        next_arm_goal = next_arm_goal.tolist()
        code = self._arm.set_servo_angle_j(angles=next_arm_goal, is_radian=True, wait=False)

        if not self._check_code(code, "set_servo_angle_j"):
            raise ValueError("position control error")
        if ignore_error:
            self._arm.clean_error()
            self._arm.clean_warn()
            
    # ======= xarm control END =======
    def enforce_z_down(self, fk: np.ndarray) -> np.ndarray:
        """Modify forward kinematics transform to enforce z-axis pointing down while preserving yaw. Not used externally (commented out in run method)."""
        fk = fk.copy()
        R = fk[:3, :3]

        # Desired z axis in base frame (pointing down)
        z_axis = np.array([0.0, 0.0, -1.0])

        # Take current x-axis of the EE in base frame
        x_curr = R[:, 0]

        # Project x_curr onto plane orthogonal to z_axis
        x_proj = x_curr - np.dot(x_curr, z_axis) * z_axis
        norm = np.linalg.norm(x_proj)
        if norm < 1e-6:
            # If projection is degenerate, fall back to a default horizontal axis
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = x_proj / norm

        # y = z × x to complete right-handed frame
        y_axis = np.cross(z_axis, x_axis)

        # Orthonormalize y just in case
        y_axis /= np.linalg.norm(y_axis)

        R_new = np.column_stack([x_axis, y_axis, z_axis])
        fk[:3, :3] = R_new
        return fk
    
    def run(self):
        """Main process loop: initialize robot, start position update thread, receive commands via UDP, and execute control."""
        # the arm initialization must be invoked in the same process
        print("Connecting to xarm at", self._ip)
        self._arm = XArmAPI(self._ip)
        print("Connected to xarm at", self._ip)
        if self.verbose:
            self.log(f"Connected to xarm at {self._ip}")
        self._init_robot()

        # NOTE: the arm can only be interacted with the process created the API class, 
        self.update_pos_t = threading.Thread(target=self.update_cur_position, name="update_cur_position")
        self.update_pos_t.start()

        port = XARM_CONTROL_PORT_L if self.robot_id <= 0 else XARM_CONTROL_PORT_R  # -1: keyboard, 0: gello left, 1: gello right
        self.command_receiver = udpReceiver(ports={'xarm_control': port})
        self.command_receiver.start()

        print(f"{'='*20} xarm reset! state: {ControllerState(self.state.value)}")
        
        command = None
        self.teleop_activated.value = False #if self.control_mode == 'joints' else True
        exe_rate = Rate(
            duration=1/self.COMMAND_EXECUTION_FREQ,
        )
        comm_rate = Rate(
            duration=1/self.comm_update_fps,
        )

        # NOTE: runs at command update freq
        while self.state.value == ControllerState.RUNNING.value:
            try:
                outer_start = time.time()
                commands = self.command_receiver.get("xarm_control", pop=True)
                if commands is None or len(commands[0]) == 0:
                    # zero action case - fast loop
                    if self.control_mode == 'velocity_control':
                        self._arm.vc_set_joint_velocity([0, 0, 0, 0, 0, 0, 0], is_radian=True, is_sync=False, duration=0)
                    elif self.control_mode == 'position_control':
                        self._arm.set_servo_angle_j(angles=self._arm.get_servo_angle(is_radian=True)[1][:7], is_radian=True, speed=1.0, wait=False)
                    exe_rate.sleep()
                    continue

                # if len(commands[0]) > 0:
                #     print(f'activated: {self.teleop_activated.value}, commands: {[np.round(c, 4) for c in commands[0]]}')

                with self.exe_lock:
                    self.log("xarm controller get lock")
                    self.log(f"new commands at {time.time() - self.start_time:.2f}")
                    
                    # NOTE: each robot command is a element in commands
                    # NOTE: command always teleop device in joint space
                    for command in commands:
                        self.log(f"get command: {command}")

                        command_goal = np.array(command)
                        assert len(command_goal) == 8, "received gello-xarm qpos must be 8-dim"

                        arm_state_goal = command_goal[0:7]  # (7,)
                        gripper_goal = command_goal[-1] if self.gripper_enable else None

                        curr_arm_state = np.array(self._arm.get_servo_angle()[1][0:7]) / 180. * np.pi
                        prev_arm_state = curr_arm_state

                        # check teleop activated and safety
                        next_arm_state = self.preprocess_command(arm_state_goal=arm_state_goal, curr_arm_state=curr_arm_state)

                        self.position_control(
                            next_arm_goal=next_arm_state, 
                            prev_arm_goal=prev_arm_state, 
                            next_gripper=gripper_goal,
                            ema_factor=self.ema_factor
                        )

                        # while time.time() - outer_start < 1.0 / self.comm_update_fps: # while didn't pass 
                        #     inner_t = time.time()
                        #     alpha = np.clip((inner_t - outer_start) * self.comm_update_fps, 0, 1)

                        #     interp_next_state = self.interpolate(curr_arm_state, next_arm_state, alpha) # interpolate in joint space with shortest arc

                        #     # communicate to low-level control
                        #     if self.control_mode == 'position_control':
                        #         # print("curr arm state: ", np.round(curr_arm_state,4))
                        #         self.position_control(
                        #             next_arm_goal=interp_next_state, 
                        #             prev_arm_goal=prev_arm_state, 
                        #             next_gripper=gripper_goal,
                        #             ema_factor=self.ema_factor)
                        #         prev_arm_state = interp_next_state
                        #     elif self.control_mode == 'velocity_control':
                        #         raise NotImplementedError("velocity control with interpolation not supported yet")
                        #         self.velocity_control(
                        #             next_arm_state, 
                        #             curr_arm_state, 
                        #             self.ema_factor)

                        #     exe_rate.sleep()
                            # print(f"xarm inner loop exe freq: {1/(time.time()-inner_t):.2f} Hz") # ~50 hz

                    if command == "quit":
                        break

                comm_rate.sleep()
                # print(f"xarm outer loop comm freq: {1/(time.time()-outer_start):.2f} Hz") # ~15 Hz

            except BaseException as e:
                self.log(f"Error in xarm controller: {e.with_traceback()}")
                break

        self.stop()
        self.command_receiver.stop()
        print("xarm controller stopped")

    def start(self) -> None:
        """Start the controller process."""
        return super().start()
    
    def stop(self):
        """Stop the controller: disable force sensor, disconnect robot, and set state to STOP."""
        self.state.value = ControllerState.STOP.value
        self._arm.set_ft_sensor_mode(0)
        self._arm.set_ft_sensor_enable(0)
        self._arm.disconnect()
        print(f"{'='*20} xarm exit! state: {ControllerState(self.state.value)}")
        # self.log(f"{'='*20} xarm exit! state: {ControllerState(self.state.value)}")

    @property
    def is_alive(self):
        """check whether the robot and the controller is alive
        To check only the controller condition, use self.state.value == ControllerState.RUNNING.value"""
        if self.is_controller_alive and self._arm.connected and self._arm.error_code == 0:
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    @property
    def is_controller_alive(self):
        return self.state.value == ControllerState.RUNNING.value

    
if __name__ == "__main__":

    start_time = time.time()
    controller = XarmController(
        start_time=start_time,
        gripper_enable=False,
        z_plane_height=434,
        verbose=True,)
    controller.start()
    controller.join()
