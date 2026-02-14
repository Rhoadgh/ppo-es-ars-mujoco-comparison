import gymnasium as gym
import tensorflow as tf
import numpy as np
import h5py
import os
import json
from es_distributed.policies import MujocoPolicy

# --- CONFIGURATION ---
LOGS_BASE_DIR = "./logs"
VIDEOS_BASE_DIR = "./manual_videos"


def record_latest():
    # 1. Find the latest experiment folder
    all_exps = [os.path.join(LOGS_BASE_DIR, d) for d in os.listdir(LOGS_BASE_DIR)
                if os.path.isdir(os.path.join(LOGS_BASE_DIR, d))]
    if not all_exps:
        print("❌ No experiments found in logs!")
        return

    latest_exp_dir = max(all_exps, key=os.path.getctime)
    exp_name = os.path.basename(latest_exp_dir)
    print(f"📂 Checking latest experiment: {exp_name}")

    # 2. Find the latest .h5 file in that folder
    snapshots = [os.path.join(latest_exp_dir, s) for s in os.listdir(latest_exp_dir) if s.endswith('.h5')]
    if not snapshots:
        print("❌ No snapshots (.h5 files) found yet!")
        return

    latest_snapshot = max(snapshots, key=os.path.getctime)
    iter_label = os.path.basename(latest_snapshot).replace(".h5", "")

    # 3. Load config for Env ID and Policy Args
    config_path = os.path.join(latest_exp_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
        env_id = config['env_id']
        policy_args = config['policy']['args']

    # 4. Prepare Video Folder
    video_folder = os.path.join(VIDEOS_BASE_DIR, exp_name)
    if not os.path.exists(video_folder): os.makedirs(video_folder)

    print(f"🎥 Recording LATEST snapshot: {iter_label} for {env_id}...")

    # 5. Gym and TF Setup
    tf.compat.v1.reset_default_graph()
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=video_folder, name_prefix=f"manual_{iter_label}")

    with tf.compat.v1.Session() as sess:
        policy = MujocoPolicy(env.observation_space, env.action_space, **policy_args)
        sess.run(tf.compat.v1.global_variables_initializer())

        # Load Weights
        with h5py.File(latest_snapshot, 'r') as f:
            group = f['MujocoPolicy']
            if 'ob_mean:0' in group:
                policy.set_ob_stat(group['ob_mean:0'][:], group['ob_std:0'][:])

            flat_components = []
            for layer_name in ['l0', 'l1', 'out']:
                lg = group[layer_name]
                flat_components.append(lg['w:0'][:].flatten())
                flat_components.append(lg['b:0'][:].flatten())
            policy.set_trainable_flat(np.concatenate(flat_components))

        # Run Episode
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0
        while not (done or truncated):
            action = policy.act(obs[None, :], None)
            if isinstance(action, (list, np.ndarray)): action = action[0]
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

    env.close()
    print(f"✅ Finished! Video saved to: {video_folder}")
    print(f"📊 Reward for this run: {total_reward:.2f}")


if __name__ == "__main__":
    tf.compat.v1.disable_v2_behavior()
    record_latest()