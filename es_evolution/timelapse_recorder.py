import gymnasium as gym
import tensorflow as tf
import numpy as np
import h5py
import os
import time
from es_distributed.policies import MujocoPolicy

# --- CONFIGURATION ---
LOG_DIR = "./logs"
VIDEO_DIR = "./videos"
ENV_ID = "Hopper-v4"
POLICY_ARGS = {
    'ac_bins': 'uniform:10',
    'ac_noise_std': 0.01,
    'connection_type': 'ff',
    'hidden_dims': [64, 64],
    'nonlin_type': 'tanh'
}


def record_snapshot(snapshot_path, iter_name):
    print(f"\n🎥 Recording new snapshot: {iter_name}")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=VIDEO_DIR, name_prefix=iter_name)

    tf.compat.v1.reset_default_graph()  # Clear graph for new load
    sess = tf.compat.v1.InteractiveSession()
    policy = MujocoPolicy(env.observation_space, env.action_space, **POLICY_ARGS)
    sess.run(tf.compat.v1.global_variables_initializer())

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

    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action = policy.act(obs[None, :], None)
        if isinstance(action, (list, np.ndarray)): action = action[0]
        obs, reward, done, truncated, info = env.step(action)

    env.close()
    sess.close()
    print(f"✅ Video saved for {iter_name}")


def monitor():
    if not os.path.exists(VIDEO_DIR): os.makedirs(VIDEO_DIR)
    processed_files = set()
    print(f"👀 Watching {LOG_DIR} for new snapshots...")

    while True:
        # Get all .h5 files
        files = [f for f in os.listdir(LOG_DIR) if f.endswith('.h5')]
        for f in sorted(files):
            if f not in processed_files:
                path = os.path.join(LOG_DIR, f)
                iter_label = f.replace(".h5", "")
                try:
                    record_snapshot(path, iter_label)
                    processed_files.add(f)
                except Exception as e:
                    print(f"Skipping {f} for now, might be currently writing... ({e})")

        time.sleep(10)  # Wait 10 seconds before checking again


if __name__ == "__main__":
    tf.compat.v1.disable_v2_behavior()
    monitor()