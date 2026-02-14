'''
Parallel implementation of the Augmented Random Search method.
Fixed: "Toggle Switch" recording to prevent video floods.
Schedule: Records approx 5 videos (Beginning, Middle, End).
'''

import argparse
import time
import os
import numpy as np
import gymnasium as gym
import logz
import ray
import utils
import optimizers
import gc
from policies import *
from shared_noise import *
from tensorboardX import SummaryWriter
from gymnasium.wrappers import RecordVideo
from datetime import datetime

@ray.remote
class Worker(object):
    def __init__(self, env_seed, env_name='', policy_params=None, deltas=None, rollout_length=1000, delta_std=0.02, record_video=False, logdir='data'):
        # 1. Use 'rgb_array' for recording, None for standard workers
        render_mode = "rgb_array" if record_video else None
        self.env = gym.make(env_name, render_mode=render_mode)

        # THE TOGGLE SWITCH
        self.allow_recording = False

        # 2. Wrap with RecordVideo if this is the designated recorder
        if record_video:
            self.env = RecordVideo(
                self.env,
                video_folder=os.path.join(logdir, "videos"),
                # CRITICAL FIX: Only record if the switch is ON
                episode_trigger=lambda x: self.allow_recording,
                name_prefix="ars_eval"
            )

        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        else:
            raise NotImplementedError
        self.delta_std = delta_std
        self.rollout_length = rollout_length

    def rollout(self, shift=0., rollout_length=None):
        if rollout_length is None:
            rollout_length = self.rollout_length
        total_reward = 0.
        steps = 0

        ob, _ = self.env.reset()
        for i in range(rollout_length):
            action = self.policy.act(ob)
            ob, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            steps += 1
            total_reward += (reward - shift)
            if done: break
        return total_reward, steps

    def do_rollouts(self, w_policy, num_rollouts=1, shift=1, evaluate=False, record_this_now=False):
        rollout_rewards, deltas_idx = [], []
        steps = 0

        for i in range(num_rollouts):
            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                self.policy.update_filter = False
                max_steps = self.env.spec.max_episode_steps if self.env.spec.max_episode_steps else 1000

                # --- VIDEO TOGGLE LOGIC ---
                if record_this_now:
                    self.allow_recording = True  # Flip switch ON

                # Run the rollout (Recorder checks switch -> Starts if True)
                reward, r_steps = self.rollout(shift=0., rollout_length=max_steps)

                if record_this_now:
                    self.allow_recording = False # Flip switch OFF immediately
                # --------------------------

                rollout_rewards.append(reward)
            else:
                idx, delta = self.deltas.get_delta(w_policy.size)
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)
                self.policy.update_filter = True
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps = self.rollout(shift=shift)
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps = self.rollout(shift=shift)
                steps += pos_steps + neg_steps
                rollout_rewards.append([pos_reward, neg_reward])

        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps": steps}

    def stats_increment(self):
        self.policy.observation_filter.stats_increment()

    def get_weights(self):
        return self.policy.get_weights()

    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)

    def get_weights_plus_stats(self):
        return self.policy.get_weights_plus_stats()


class ARSLearner(object):
    def __init__(self, env_name='HalfCheetah-v4', policy_params=None, num_workers=32, num_deltas=320, deltas_used=320,
                 delta_std=0.02, logdir=None, rollout_length=1000, step_size=0.01, shift=0, params=None, seed=123, record=False):
        logz.configure_output_dir(logdir)
        logz.save_params(params)
        env = gym.make(env_name)
        self.timesteps = 0
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.num_workers = num_workers
        self.record = record

        run_name = f"{env_name}__ARS__{seed}__{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{run_name}")

        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed=seed + 3)

        self.workers = []
        for i in range(num_workers):
            # Record only if the global record flag is True AND it's Worker 0
            is_recorder = (i == 0 and self.record)

            w = Worker.remote(
                seed + 7 * i,
                env_name=env_name,
                policy_params=policy_params,
                deltas=deltas_id,
                rollout_length=rollout_length,
                delta_std=delta_std,
                record_video=is_recorder,
                logdir=self.logdir
            )
            self.workers.append(w)

        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)

    def aggregate_rollouts(self, num_rollouts=None, evaluate=False):
        num_deltas = self.num_deltas if num_rollouts is None else num_rollouts
        policy_id = ray.put(self.w_policy)
        num_rollouts_per_worker = int(num_deltas / self.num_workers)

        rollout_ids = [worker.do_rollouts.remote(policy_id, num_rollouts=num_rollouts_per_worker, shift=self.shift,
                                                 evaluate=evaluate) for worker in self.workers]
        if num_deltas % self.num_workers > 0:
            rollout_ids += [worker.do_rollouts.remote(policy_id, num_rollouts=1, shift=self.shift, evaluate=evaluate)
                            for worker in self.workers[:(num_deltas % self.num_workers)]]

        results = ray.get(rollout_ids)
        rollout_rewards, deltas_idx = [], []
        for result in results:
            if not evaluate: self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        rollout_rewards = np.array(rollout_rewards, dtype=np.float64)
        if evaluate: return rollout_rewards

        max_rewards = np.max(rollout_rewards, axis=1)
        idx = np.arange(max_rewards.size)[
            max_rewards >= np.percentile(max_rewards, 100 * (1 - (self.deltas_used / self.num_deltas)))]

        deltas_idx = np.array(deltas_idx)[idx]
        rollout_rewards = rollout_rewards[idx, :]
        rollout_rewards /= (np.std(rollout_rewards) + 1e-8)

        g_hat, _ = utils.batched_weighted_sum(rollout_rewards[:, 0] - rollout_rewards[:, 1],
                                              (self.deltas.get(d_idx, self.w_policy.size) for d_idx in deltas_idx),
                                              batch_size=500)
        return g_hat / deltas_idx.size

    def train(self, total_timesteps=3000000):
        iteration = 0
        start = time.time()

        try:
            while self.timesteps < total_timesteps:
                iteration += 1

                # 1. Sync weights for this iteration
                policy_id = ray.put(self.w_policy)

                # 2. VIDEO RECORDING (Schedule: 1, 20, 40, 60, 80...)
                # This gives you "Beginning, Middle, End" without flooding the folder
                if (iteration == 1 or iteration % 20 == 0):
                    print(f"--> Recording milestone video at iteration {iteration}")
                    ray.get(self.workers[0].do_rollouts.remote(policy_id, num_rollouts=1,
                                                               evaluate=True, record_this_now=True))

                # 3. Calculate Gradient & Update
                g_hat = self.aggregate_rollouts()
                self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)

                # 4. Save Weights
                w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                np.savez(self.logdir + "/lin_policy_plus", w)

                # 5. LOGGING (Every 10 Iterations)
                if (iteration % 10 == 0):
                    # These 100 rollouts will NOT trigger video anymore!
                    rewards = self.aggregate_rollouts(num_rollouts=100, evaluate=True)

                    avg_reward = np.mean(rewards)
                    current_time = time.time() - start
                    sps = int(self.timesteps / current_time) if current_time > 0 else 0

                    logz.log_tabular("Iteration", iteration)
                    logz.log_tabular("Total_Timesteps", self.timesteps)
                    logz.log_tabular("AverageReward", avg_reward)
                    logz.log_tabular("Time", current_time)
                    logz.dump_tabular()

                    self.writer.add_scalar("charts/episodic_return", avg_reward, self.timesteps)
                    self.writer.add_scalar("charts/SPS", sps, self.timesteps)

                # 6. Housekeeping
                for j in range(self.num_workers):
                    self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
                self.policy.observation_filter.stats_increment()
                self.policy.observation_filter.clear_buffer()

                filter_id = ray.put(self.policy.observation_filter)
                ray.get([worker.sync_filter.remote(filter_id) for worker in self.workers])
                ray.get([worker.stats_increment.remote() for worker in self.workers])

                if iteration % 50 == 0:
                    gc.collect()

        finally:
            print(f"\nTarget of {total_timesteps} reached or interrupted.")
            print("Finalizing and saving weights...")
            w = ray.get(self.workers[0].get_weights_plus_stats.remote())
            np.savez(self.logdir + "/lin_policy_plus", w)
            self.writer.close()


def run_ars(params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"{params['env_name']}__w{params['n_workers']}__s{params['seed']}__{timestamp}"
    run_logdir = os.path.join(params['dir_path'], folder_name)

    video_dir = os.path.join(run_logdir, "videos")
    if os.path.exists(video_dir):
        import shutil
        shutil.rmtree(video_dir)  # Deletes old videos from previous failed attempts
    os.makedirs(video_dir)

    if not os.path.exists(run_logdir):
        os.makedirs(run_logdir)

    print(f"Saving all results to: {run_logdir}")

    env = gym.make(params['env_name'])
    ob_dim, ac_dim = env.observation_space.shape[0], env.action_space.shape[0]
    policy_params = {'type': 'linear', 'ob_filter': params['filter'], 'ob_dim': ob_dim, 'ac_dim': ac_dim}

    ARS = ARSLearner(
        env_name=params['env_name'],
        policy_params=policy_params,
        num_workers=params['n_workers'],
        num_deltas=params['n_directions'],
        deltas_used=params['deltas_used'],
        step_size=params['step_size'],
        delta_std=params['delta_std'],
        logdir=run_logdir,
        rollout_length=params['rollout_length'],
        shift=params['shift'],
        params=params,
        seed=params['seed'],
        record=params.get('record', False)
    )

    ARS.train(total_timesteps=params['total_timesteps'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--step_size', type=float, default=0.03)
    parser.add_argument('--delta_std', type=float, default=0.1)
    parser.add_argument('--n_directions', type=int, default=25)
    parser.add_argument('--deltas_used', type=int, default=25)
    parser.add_argument('--rollout_length', type=int, default=1000)
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--dir_path', type=str, default='data')
    parser.add_argument('--filter', type=str, default='MeanStdFilter')
    parser.add_argument('--record', action='store_true', help='Enable video recording on Worker 0')
    parser.add_argument('--total_timesteps', type=int, default=3000000)

    args = parser.parse_args()

    # Initialize Ray
    ray.init(num_cpus=args.n_workers, ignore_reinit_error=True)

    # Run the training
    run_ars(vars(args))