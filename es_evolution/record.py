import gymnasium as gym
import tensorflow as tf
import numpy as np
import h5py
import os
import sys
import json
from es_distributed.policies import MujocoPolicy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ---------------------------------
# Require snapshot path from CLI
# ---------------------------------
if len(sys.argv) < 2:
    raise ValueError("Usage: python record.py <snapshot_path>")

SNAPSHOT_PATH = sys.argv[1]
EXP_DIR = os.path.dirname(SNAPSHOT_PATH)

# ---------------------------------
# Detect environment from folder name
# ---------------------------------
folder_name = os.path.basename(EXP_DIR).lower()

if "hopper" in folder_name:
    config_file = "hopper.json"
    ENV_ID = "Hopper-v4"
elif "cheetah" in folder_name:
    config_file = "cheetah.json"
    ENV_ID = "HalfCheetah-v4"
elif "ant" in folder_name:
    config_file = "Ant.json"
    ENV_ID = "Ant-v4"
elif "walker" in folder_name:
    config_file = "Walker2D.json"
    ENV_ID = "Walker2d-v5"
else:
    raise ValueError("Could not detect environment from folder name")

print(f"Detected environment: {ENV_ID}")
print(f"Using config file: {config_file}")

# ---------------------------------
# Load JSON config
# ---------------------------------
# -------------------------------
# Load JSON config
# -------------------------------
CONFIG_PATH = os.path.join(BASE_DIR, "configurations", config_file)
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

ENV_ID = config["env_id"]
POLICY_ARGS = config["policy"]["args"]

print(f"Using environment: {ENV_ID}")
print(f"Policy arguments: {POLICY_ARGS}")

def record():
    # 1. Setup Environment with Video Recording
    print("Initializing Environment...")
    # 'rgb_array' mode allows recording without a live window popping up
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="./videos",
        name_prefix=ENV_ID.replace("-", "_")
    )
    env = gym.wrappers.RecordVideo(env, video_folder="./videos", name_prefix="hopper_eval")

    # 2. Setup TF
    tf.compat.v1.disable_v2_behavior()
    sess = tf.compat.v1.InteractiveSession()
    policy = MujocoPolicy(env.observation_space, env.action_space, **POLICY_ARGS)
    sess.run(tf.compat.v1.global_variables_initializer())

    # 3. Load Weights
    print(f"Loading {SNAPSHOT_PATH}...")
    with h5py.File(SNAPSHOT_PATH, 'r') as f:
        group = f['MujocoPolicy']
        if 'ob_mean:0' in group:
            policy.set_ob_stat(group['ob_mean:0'][:], group['ob_std:0'][:])

        flat_components = []
        for layer_name in ['l0', 'l1', 'out']:
            lg = group[layer_name]
            flat_components.append(lg['w:0'][:].flatten())
            flat_components.append(lg['b:0'][:].flatten())
        policy.set_trainable_flat(np.concatenate(flat_components))

        # 4. Run one episode to record it
        print("Recording episode... please wait...")
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            try:
                # Your policy only returns the action, so we don't use 'action, _'
                action = policy.act(obs[None, :], None)

                # If action comes back as a list or nested array, take the first element
                if isinstance(action, (list, np.ndarray)):
                    action = action[0]
            except Exception as e:
                # Final fallback if act() fails
                action = policy.get_action(obs)

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        print(f"Recording Complete! Reward: {total_reward}")
        env.close()
        print("Check the './videos' folder for the mp4 file.")

if __name__ == "__main__":
    record()