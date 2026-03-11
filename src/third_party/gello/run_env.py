import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict
import argparse
import numpy as np
import pickle
import termcolor

from third_party.gello.agents.agent import BimanualAgent, DummyAgent
from third_party.gello.agents.gello_agent import GelloAgent
from third_party.gello.env import RobotEnv
from third_party.gello.zmq_core.robot_node import ZMQClientRobot
from third_party.gello.agents.gello_agent import DynamixelRobotConfig


def save_frame(
    folder: Path,
    timestamp: datetime.datetime,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
) -> None:
    obs["control"] = action  # add action to obs

    # make folder if it doesn't exist
    folder.mkdir(exist_ok=True, parents=True)
    recorded_file = folder / (timestamp.isoformat() + ".pkl")

    with open(recorded_file, "wb") as f:
        pickle.dump(obs, f)

def print_color(*args, color=None, attrs=(), **kwargs):
    

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


def main(args):

    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict={})

    if args.bimanual:
        dynamixel_config_left = DynamixelRobotConfig(
            joint_ids=(1, 2, 3, 4, 5, 6, 7),
            joint_offsets=(
                1 * np.pi / 2,
                2 * np.pi / 2,
                0 * np.pi / 2,
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
                1 * np.pi / 2,
                2 * np.pi / 2,
                0 * np.pi / 2,
                1 * np.pi / 2,
                2 * np.pi / 2,
                2 * np.pi / 2,
                2 * np.pi / 2
            ),
            joint_signs=(1, 1, 1, 1, 1, 1, 1),
            gripper_config=(8, 288, 246),
        )
        right = args.right_gello_port
        left = args.left_gello_port
        if args.right_start_joints is None:
            right_reset_joints = np.deg2rad(
                [0, -45, 0, 30, 0, 75, 0, 0]
            ) # Change this to your own reset joints
        else:
            right_reset_joints = args.right_start_joints
        if args.left_start_joints is None:
            left_reset_joints = np.deg2rad(
                [0, -45, 0, 30, 0, 75, 0, 0]
            ) # Change this to your own reset joints
        else:
            left_reset_joints = args.left_start_joints
        reset_joints = np.concatenate([left_reset_joints, right_reset_joints])
        left_agent = GelloAgent(port=left, dynamixel_config=dynamixel_config_left, start_joints=args.left_start_joints)
        right_agent = GelloAgent(port=right, dynamixel_config=dynamixel_config_right, start_joints=args.right_start_joints)
        agent = BimanualAgent(left_agent, right_agent)

    else:
        dynamixel_config = DynamixelRobotConfig(
            joint_ids=(1, 2, 3, 4, 5, 6, 7),
            joint_offsets=(
                3 * np.pi / 2,
                1 * np.pi / 2,
                3 * np.pi / 2,
                1 * np.pi / 2,
                4 * np.pi / 2,
                1 * np.pi / 2,
                2 * np.pi / 2
            ),
            joint_signs=(1, 1, 1, 1, 1, 1, 1),
            gripper_config=(8, 127, 87),
        )
        gello_port = args.gello_port
        if args.start_joints is None:
            reset_joints = np.deg2rad(
                [0, -45, 0, 30, 0, 75, 0, 0]
            ) # Change this to your own reset joints
        else:
            reset_joints = args.start_joints
        agent = GelloAgent(port=gello_port, dynamixel_config=dynamixel_config, start_joints=args.start_joints)
    
    curr_joints = env.get_obs()["joint_positions"]
    if reset_joints.shape == curr_joints.shape:
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)
        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
            time.sleep(0.001)
    

    # going to start position
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.5).any():
        print("Action is too big")

        # print which joints are too big
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    start_time = time.time()
    while True:
        num = time.time() - start_time
        message = f"\rTime passed: {round(num, 2)}          "
        print_color(
            message,
            color="white",
            attrs=("bold",),
            end="",
            flush=True,
        )
        action = agent.act(obs)
        obs = env.step(action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bimanual", action="store_true")
    parser.add_argument("--robot_port", type=int, default=6001)
    parser.add_argument("--hostname", type=str, default="127.0.0.2")
    parser.add_argument("--hz", type=int, default=100)
    parser.add_argument("--start_joints", type=float, nargs="+", default=None)
    parser.add_argument("--right_start_joints", type=float, nargs="+", default=None)
    parser.add_argument("--left_start_joints", type=float, nargs="+", default=None)
    parser.add_argument("--gello_port", type=str, default=None)
    parser.add_argument("--right_gello_port", type=str, default=None)
    parser.add_argument("--left_gello_port", type=str, default=None)
    args = parser.parse_args()
    main(args)
