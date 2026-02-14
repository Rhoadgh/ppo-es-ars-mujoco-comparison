import gymnasium as gym
import tensorflow as tf
import numpy as np
import h5py
from es_distributed.policies import MujocoPolicy
import time

# --- CONFIGURATION ---
# Use the latest snapshot in your ./logs folder
SNAPSHOT_PATH = "./logs/snapshot_iter00040_rew2394.h5"

ENV_ID = "Hopper-v4"

# These must match your hopper.json exactly
POLICY_ARGS = {
    'ac_bins': 'uniform:10',
    'ac_noise_std': 0.01,
    'connection_type': 'ff',
    'hidden_dims': [64, 64],
    'nonlin_type': 'tanh'
}


def load_and_visualize():
    # 1. Setup Environment
    # We use "human" mode to actually see the window
    env = gym.make(ENV_ID, render_mode="human")

    # 2. Setup TensorFlow 1.x Compatibility
    tf1 = tf.compat.v1
    tf1.disable_v2_behavior()
    sess = tf1.InteractiveSession()

    # 3. Initialize Policy
    policy = MujocoPolicy(env.observation_space, env.action_space, **POLICY_ARGS)
    sess.run(tf1.global_variables_initializer())

    # 4. Load the Weights
    print(f"Loading weights from {SNAPSHOT_PATH}...")
    try:
        with h5py.File(SNAPSHOT_PATH, 'r') as f:
            group = f['MujocoPolicy']

            # 1. Load Observation Normalization (Crucial for the robot to 'see' correctly)
            if 'ob_mean:0' in group and 'ob_std:0' in group:
                print("Loading observation normalization constants...")
                policy.set_ob_stat(group['ob_mean:0'][:], group['ob_std:0'][:])

            # 2. Reconstruct the flat weights vector
            # The order must be exactly: l0/w, l0/b, l1/w, l1/b, out/w, out/b
            flat_components = []
            for layer_name in ['l0', 'l1', 'out']:
                # H5 stores these as further sub-groups
                layer_group = group[layer_name]
                # Flatten weights and biases and add to our list
                flat_components.append(layer_group['w:0'][:].flatten())
                flat_components.append(layer_group['b:0'][:].flatten())

            # Combine everything into one long array
            all_weights = np.concatenate(flat_components)
            print(f"Total weights reconstructed: {len(all_weights)} parameters")

            # Feed to policy
            policy.set_trainable_flat(all_weights)
            print("Successfully loaded and set weights!")

    except Exception as e:
        print(f"Error loading snapshot: {e}")
        return

    # 5. Run the Visualizer Loop
    print("Starting visualization. Close the window or press Ctrl+C to stop.")
    while True:
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # compute_action(observation, stochastic=False)
            # We set noise to None to see the robot's "best" learned behavior
            action = policy.compute_action(obs[None, :], prec_noise=None)[0]

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode finished. Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    load_and_visualize()