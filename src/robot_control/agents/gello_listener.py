import multiprocess as mp
import time
import copy
from typing import List

import numpy as np

from third_party.gello.agents.agent import BimanualAgent
from third_party.gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig
from third_party.gello.dynamixel.driver import DynamixelDriver

np.set_printoptions(precision=2, suppress=True)


class GelloListener(mp.Process):
    def __init__(
        self, 
        # shm_manager: SharedMemoryManager, 
        bimanual: bool = False,
        gello_port: str = '/dev/ttyUSB0',
        bimanual_gello_port: List[str] = ['/dev/ttyUSB0', '/dev/ttyUSB1'],
        baudrate: int = 57600,
    ):
        super().__init__()
        
        self.bimanual = bimanual
        self.bimanual_gello_port = bimanual_gello_port

        self.num_joints = 7
        self.gello_port = gello_port
        self.baudrate = baudrate
        self.do_offset_calibration = False  # whether to recalibrate the offset
        self.verbose = True
        self.initialize_done = mp.Value('b', True)

        if bimanual:
            examples = dict()
            examples['command'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # examples['left_timestamp'] = 0.0
            # examples['right_timestamp'] = 0.0
        else:
            examples = dict()
            examples['command'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # examples['timestamp'] = 0.0

        # ring_buffer = SharedMemoryRingBuffer.create_from_examples(
        #     shm_manager=shm_manager,
        #     examples=examples,
        #     get_max_k=30,
        #     get_time_budget=0.2,
        #     put_desired_frequency=100,
        # )
        self.command = mp.Array('d', examples['command'])
        # self.ring_buffer = ring_buffer
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

    # def start(self, wait=True):
    #     super().start()
    #     if wait:
    #         self.start_wait()
    
    def stop(self, wait=False):
        self.stop_event.set()
        if wait:
            self.end_wait()

    # def start_wait(self):
    #     self.ready_event.wait()
    
    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self): # , k=None, out=None):
        return copy.deepcopy(np.array(self.command[:]))
        # if k is None:
        #     return self.ring_buffer.get(out=out)
        # else:
        #     return self.ring_buffer.get_last_k(k, out=out)

    def init_gello(self):
        if self.bimanual:
            if self.do_offset_calibration:
                assert len(self.bimanual_gello_port) == 2, "Please provide two ports for bimanual calibration"
                left_joint_offsets, left_gripper_config = self.calibrate_offset(port=self.bimanual_gello_port[0])
                right_joint_offsets, right_gripper_config = self.calibrate_offset(port=self.bimanual_gello_port[1])
                dynamixel_config_left = DynamixelRobotConfig(
                    joint_ids=(1, 2, 3, 4, 5, 6, 7),
                    joint_offsets=left_joint_offsets,
                    joint_signs=(1, 1, 1, 1, 1, 1, 1),
                    gripper_config=left_gripper_config,
                )
                dynamixel_config_right = DynamixelRobotConfig(
                    joint_ids=(1, 2, 3, 4, 5, 6, 7),
                    joint_offsets=right_joint_offsets,
                    joint_signs=(1, 1, 1, 1, 1, 1, 1),
                    gripper_config=right_gripper_config,
                )
            else:
                dynamixel_config_left = DynamixelRobotConfig(
                    joint_ids=(1, 2, 3, 4, 5, 6, 7),
                    joint_offsets=(
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        4 * np.pi / 2,
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2
                    ),
                    joint_signs=(1, 1, 1, 1, 1, 1, 1),
                    gripper_config=(8, 288, 246),
                )
                dynamixel_config_right = DynamixelRobotConfig(
                    joint_ids=(1, 2, 3, 4, 5, 6, 7),
                    joint_offsets=(
                        2 * np.pi / 2,
                        2 * np.pi / 2,
                        3 * np.pi / 2,
                        1 * np.pi / 2,
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        0 * np.pi / 2
                    ),
                    joint_signs=(1, 1, 1, 1, 1, 1, 1),
                    gripper_config=(8, 114, 72),
                )
            left_start_joints = np.deg2rad([0, -45, 0, 30, 0, 75, 0, 0])
            right_start_joints = np.deg2rad([0, -45, 0, 30, 0, 75, 0, 0])
            left_agent = GelloAgent(port=self.bimanual_gello_port[0], dynamixel_config=dynamixel_config_left, start_joints=left_start_joints)
            right_agent = GelloAgent(port=self.bimanual_gello_port[1], dynamixel_config=dynamixel_config_right, start_joints=right_start_joints)
            agent = BimanualAgent(left_agent, right_agent)
            self.agent = agent

        else:
            if self.do_offset_calibration:
                joint_offsets, gripper_config = self.calibrate_offset(port=self.gello_port)
            
            else:
                if self.gello_port == '/dev/ttyUSB1':
                    joint_offsets = (
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        4 * np.pi / 2,
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2
                    )
                    gripper_config = (8, 288, 246)
                else:
                    assert self.gello_port == '/dev/ttyUSB0'
                    joint_offsets = (
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        0 * np.pi / 2,
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2
                    )
                    gripper_config = (8, 290, 248)

            dynamixel_config = DynamixelRobotConfig(
                joint_ids=(1, 2, 3, 4, 5, 6, 7),
                joint_offsets=joint_offsets,
                joint_signs=(1, 1, 1, 1, 1, 1, 1),
                gripper_config=gripper_config,
            )
            gello_port = self.gello_port
            start_joints = np.deg2rad([0, -45, 0, 30, 0, 75, 0, 0])
            try:
                agent = GelloAgent(port=gello_port, dynamixel_config=dynamixel_config, start_joints=start_joints)
            except:
                print(f"Failed to connect to Gello on port {gello_port}")
                try:
                    agent = GelloAgent(port=self.bimanual_gello_port[1], dynamixel_config=dynamixel_config, start_joints=start_joints)
                except:
                    print(f"Failed to connect to Gello on port {gello_port} and {self.bimanual_gello_port[1]}")
                    raise
            self.agent = agent

        self.ready_event.set()
    
    def calibrate_offset(self, port, verbose=False):
        # MENAGERIE_ROOT = Path(__file__).parent / "third_party" / "mujoco_menagerie"
        
        start_joints = tuple(np.deg2rad([0, -45, 0, 30, 0, 75, 0]))  # The joint angles that the GELLO is placed in at (in radians)
        joint_signs = (1, 1, 1, 1, 1, 1, 1)  # The joint angles that the GELLO is placed in at (in radians)

        joint_ids = list(range(1, self.num_joints + 2))
        driver = DynamixelDriver(joint_ids, port=port, baudrate=self.baudrate)

        # assume that the joint state shouold be start_joints
        # find the offset, which is a multiple of np.pi/2 that minimizes the error between the current joint state and args.start_joints
        # this is done by brute force, we seach in a range of +/- 8pi

        def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
            joint_sign_i = joint_signs[index]
            joint_i = joint_sign_i * (joint_state[index] - offset)
            start_i = start_joints[index]
            return np.abs(joint_i - start_i)

        for _ in range(10):
            driver.get_joints()  # warmup

        for _ in range(1):
            best_offsets = []
            curr_joints = driver.get_joints()
            for i in range(self.num_joints):
                best_offset = 0
                best_error = 1e6
                for offset in np.linspace(
                    -8 * np.pi, 8 * np.pi, 8 * 4 + 1
                ):  # intervals of pi/2
                    error = get_error(offset, i, curr_joints)
                    if error < best_error:
                        best_error = error
                        best_offset = offset
                best_offsets.append(best_offset)

        gripper_open = np.rad2deg(driver.get_joints()[-1]) - 0.2
        gripper_close = np.rad2deg(driver.get_joints()[-1]) - 42
        if self.verbose:
            print()
            print("best offsets               : ", [f"{x:.3f}" for x in best_offsets])
            print(
                "best offsets function of pi: ["
                + ", ".join([f"{int(np.round(x/(np.pi/2)))}*np.pi/2" for x in best_offsets])
                + " ]",
            )
            print(
                "gripper open (degrees)       ",
                gripper_open,
            )
            print(
                "gripper close (degrees)      ",
                gripper_close,
            )

        joint_offsets = tuple(best_offsets)
        gripper_config = (8, gripper_open, gripper_close)
        return joint_offsets, gripper_config

    def run(self):
        self.init_gello()

        
        while self.alive:
            try:
                curr_time = time.time()
                action = self.agent.get_action()
                self.command[:] = action
                # print("gello update freq: %.2f Hz"%(1.0 / (time.time() - curr_time)))
            except:
                print(f"Error in GelloListener")
                break

        self.stop()
        print("GelloListener exit!")
        
    @property
    def alive(self):
        return not self.stop_event.is_set() and self.ready_event.is_set()


