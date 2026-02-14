import numpy as np
import gymnasium as gym

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert rollouts')
    args = parser.parse_args()

    print('loading and building expert policy')
    # Load the .npz file
    dict_data = np.load(args.expert_policy_file, allow_pickle=True)
    
    # Extract data from the first key (usually 'arr_0')
    first_key = dict_data.files[0]
    lin_policy = dict_data[first_key]
    
    # ARS saves weights, mu (mean), and std in this order
    M = lin_policy[0]
    mean = lin_policy[1]
    std = lin_policy[2]
        
    # UPDATED: Gymnasium requires render_mode to be set in make()
    render_mode = "human" if args.render else None
    env = gym.make(args.envname, render_mode=render_mode)

    # UPDATED: max_episode_steps replaces timestep_limit
    max_steps = env.spec.max_episode_steps if env.spec.max_episode_steps else 1000

    returns = []
    for i in range(args.num_rollouts):
        print('Rollout', i)
        # UPDATED: reset() returns (obs, info)
        obs, _ = env.reset()
        done = False
        totalr = 0.
        steps = 0
        
        while not done:
            # Policy math: action = Weights * normalized_observation
            action = np.dot(M, (obs - mean) / (std + 1e-8))
            
            # UPDATED: step() returns 5 values
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            totalr += r
            steps += 1
            
            # Note: env.render() is no longer needed inside the loop 
            # if render_mode="human" was used in gym.make()
            
            if steps % 100 == 0: 
                print(f"Step {steps}/{max_steps}")
            if steps >= max_steps:
                break
                
        returns.append(totalr)

    print('\n' + '='*20)
    print(f'Average Return: {np.mean(returns):.2f}')
    print(f'Std Deviation: {np.std(returns):.2f}')
    print('='*20)
    
if __name__ == '__main__':
    main()