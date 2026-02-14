import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import argparse
import os


def record_universal():
    parser = argparse.ArgumentParser()
    parser.add_argument('policy_file', type=str, help="Path to the .npz policy file")
    parser.add_argument('env_name', type=str, help="Name of the Gymnasium environment (e.g., Ant-v4)")
    parser.add_argument('--video_dir', type=str, default='./videos', help="Where to save the video")
    parser.add_argument('--num_episodes', type=int, default=1, help="Number of episodes to record")
    args = parser.parse_args()

    # 1. Load the policy data
    data = np.load(args.policy_file, allow_pickle=True)

    # Extract the main object
    if 'arr_0' in data:
        policy_data = data['arr_0']
    else:
        policy_data = data[data.files[0]]

    # Unwrap if it's a 0-d object array
    if hasattr(policy_data, 'shape') and policy_data.shape == ():
        policy_data = policy_data.item()

    # 2. Extract weights and stats based on data type (Dict vs List)
    if isinstance(policy_data, dict):
        # Format used by modern ARS saves
        weights = policy_data['weights']
        mu = policy_data['mu']
        std = policy_data['std']
    else:
        # Format used by older ARS saves (List/Array)
        weights = policy_data[0]
        mu = policy_data[1]
        std = policy_data[2]

    print(f"Successfully loaded policy. Weights shape: {weights.shape}")

    # 2. Setup Environment
    # Use "rgb_array" for recording, "human" for live viewing
    env = gym.make(args.env_name, render_mode="rgb_array")

    # Wrap for video
    env = RecordVideo(
        env,
        video_folder=args.video_dir,
        name_prefix=f"eval_{args.env_name}",
        episode_trigger=lambda x: True
    )

    for ep in range(args.num_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (terminated or truncated):
            # Abstract Policy Math: Works for any dimension
            # action = Weights * normalized_observation
            norm_obs = (obs - mu) / (std + 1e-8)
            action = np.dot(weights, norm_obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

        print(f"Episode {ep + 1} finished. Reward: {total_reward:.2f} | Steps: {steps}")

    env.close()
    print(f"\nRecording(s) saved to: {os.path.abspath(args.video_dir)}")


if __name__ == "__main__":
    record_universal()