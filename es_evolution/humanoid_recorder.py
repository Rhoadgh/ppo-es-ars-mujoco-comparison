# humanoid_recorder
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
CHECK_INTERVAL = 60  # Wait 1 minute between checks to save CPU


def record_snapshot(snapshot_path, iter_name, video_folder, env_id, policy_args):
    print(f"\n🎥 Recording LATEST snapshot: {iter_name} for {env_id}")

    tf.compat.v1.reset_default_graph()
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=video_folder, name_prefix=iter_name)

    with tf.compat.v1.Session() as sess:
        policy = MujocoPolicy(env.observation_space, env.action_space, **policy_args)
        sess.run(tf.compat.v1.global_variables_initializer())

        with h5py.File(snapshot_path, 'r') as f:
            group = f['MujocoPolicy']
            if 'ob_mean:0' in group:
                policy.set_ob_stat(group['ob_mean:0'][:], group['ob_std:0'][:])

            flat_components = []
            # Note: Humanoid [256, 256] uses l0, l1, out just like smaller nets
            for layer_name in ['l0', 'l1', 'out']:
                lg = group[layer_name]
                flat_components.append(lg['w:0'][:].flatten())
                flat_components.append(lg['b:0'][:].flatten())
            policy.set_trainable_flat(np.concatenate(flat_components))

        obs, info = env.reset()
        done = truncated = False
        total_reward = 0
        while not (done or truncated):
            action = policy.act(obs[None, :], None)
            if isinstance(action, (list, np.ndarray)): action = action[0]
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

    env.close()
    print(f"✅ Video saved. Reward: {total_reward:.2f}")


def monitor():
    last_recorded_snapshot = None
    print(f"👀 Watching {LOGS_BASE_DIR}. Will only record the NEWEST file found...")

    while True:
        all_exps = [os.path.join(LOGS_BASE_DIR, d) for d in os.listdir(LOGS_BASE_DIR)
                    if os.path.isdir(os.path.join(LOGS_BASE_DIR, d))]

        if not all_exps:
            time.sleep(CHECK_INTERVAL)
            continue

        latest_exp_dir = max(all_exps, key=os.path.getctime)
        exp_name = os.path.basename(latest_exp_dir)
        exp_video_dir = os.path.join(VIDEOS_BASE_DIR, exp_name)
        if not os.path.exists(exp_video_dir): os.makedirs(exp_video_dir)

        config_path = os.path.join(latest_exp_dir, "config.json")
        if not os.path.exists(config_path):
            time.sleep(5)
            continue

        with open(config_path, 'r') as f:
            config = json.load(f)
            env_id = config['env_id']
            policy_args = config['policy']['args']

        # --- THE FIX: FIND ONLY THE ABSOLUTE LATEST FILE ---
        snapshots = [os.path.join(latest_exp_dir, s) for s in os.listdir(latest_exp_dir) if s.endswith('.h5')]
        if snapshots:
            newest_snapshot = max(snapshots, key=os.path.getctime)

            # Only record if this is a DIFFERENT file than the one we last recorded
            if newest_snapshot != last_recorded_snapshot:
                iter_label = os.path.basename(newest_snapshot).replace(".h5", "")
                try:
                    record_snapshot(newest_snapshot, iter_label, exp_video_dir, env_id, policy_args)
                    last_recorded_snapshot = newest_snapshot
                except Exception as e:
                    print(f"⌛ File busy, retrying later: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    tf.compat.v1.disable_v2_behavior()
    monitor()