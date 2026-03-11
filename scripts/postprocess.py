import argparse

from robot_control.utils.postprocesser import synchronize_timesteps, pack_episode_trajectories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to recording root directory')
    parser.add_argument('--pack', action='store_true', help='also pack episode trajectories into .npy')
    args = parser.parse_args()

    assert args.data_path != '', "Please provide a path to the data"

    synchronize_timesteps(data_path=args.data_path)

    if args.pack:
        for method in ("teleop", "residual_copilot", "residual_bc"):
            pack_episode_trajectories(data_path=args.data_path, method=method)
