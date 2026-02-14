import gymnasium as gym
import tensorflow as tf
import numpy as np
import h5py
import os
import time
import json
from es_distributed.policies import MujocoPolicy

# --- CONFIGURATION ---
LOGS_BASE_DIR = "./logs"
VIDEOS_BASE_DIR = "./videos"


def record_snapshot(snapshot_path, iter_name, video_folder, env_id, policy_args):
    print(f"\n🎥 Recording snapshot: {iter_name} for {env_id}")

    # 1. Clear everything from the previous run
    tf.compat.v1.reset_default_graph()

    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=video_folder, name_prefix=iter_name)

    # 2. Use a standard Session instead of InteractiveSession
    with tf.compat.v1.Session() as sess:
        policy = MujocoPolicy(env.observation_space, env.action_space, **policy_args)
        sess.run(tf.compat.v1.global_variables_initializer())

        # Load Weights
        with h5py.File(snapshot_path, 'r') as f:
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

    # Session automatically closes here because of the 'with' block
    env.close()
    print(f"✅ Video saved. Reward: {total_reward:.2f}")


def monitor():
    processed_snapshots = set()
    print(f"👀 Watching {LOGS_BASE_DIR} for the latest experiment snapshots...")

    while True:
        # 1. Identify the latest experiment folder
        all_exps = [os.path.join(LOGS_BASE_DIR, d) for d in os.listdir(LOGS_BASE_DIR)
                    if os.path.isdir(os.path.join(LOGS_BASE_DIR, d))]

        if not all_exps:
            time.sleep(10)
            continue

        latest_exp_dir = max(all_exps, key=os.path.getctime)
        exp_name = os.path.basename(latest_exp_dir)

        # 2. Create a dedicated video subfolder for this experiment
        exp_video_dir = os.path.join(VIDEOS_BASE_DIR, exp_name)
        if not os.path.exists(exp_video_dir): os.makedirs(exp_video_dir)

        # 3. Load the config.json for this specific experiment
        config_path = os.path.join(latest_exp_dir, "config.json")
        if not os.path.exists(config_path):
            time.sleep(5)
            continue

        with open(config_path, 'r') as f:
            config = json.load(f)
            env_id = config['env_id']
            policy_args = config['policy']['args']

        # 4. Check for new .h5 snapshots in the latest experiment folder
        snapshots = [s for s in os.listdir(latest_exp_dir) if s.endswith('.h5')]
        for s in sorted(snapshots):
            unique_id = os.path.join(exp_name, s)  # Track by folder + filename
            if unique_id not in processed_snapshots:
                snapshot_path = os.path.join(latest_exp_dir, s)
                iter_label = s.replace(".h5", "")

                try:
                    record_snapshot(snapshot_path, iter_label, exp_video_dir, env_id, policy_args)
                    processed_snapshots.add(unique_id)
                except Exception as e:
                    print(f"⌛ Skipping {s} (file might be busy): {e}")

        time.sleep(10)


if __name__ == "__main__":
    tf.compat.v1.disable_v2_behavior()
    monitor()