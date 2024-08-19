# Import the registered environment
import CustEnv

# Import the grid loader
from grid_loader import load_test_case_grid

# Import the gym
import gymnasium as gym

# Import stable baselines stuff
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Import other stuff
import matplotlib.pyplot as plt



# Load the trained model
model = PPO.load("Training/Model/Case14_EV")

def evaluate_PPO_model(n_case=14, model=None):
    # Load the environment
    grid = load_test_case_grid(n_case)
    eval_env = gym.make('PowerGrid-v0', net=grid, dispatching_intervals=24, EVaware=True)

    # Initialize variables for tracking metrics
    n_steps = 24  
    rewards = []
    # actions = []
    # observations = []
    infos = []
    truncated = False
    terminated = False

    # Evaluate the model
    obs, info = eval_env.reset()
    for step in range(n_steps):
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        # Log the metrics
        rewards.append(reward)
        # actions.append(action)
        # observations.append(obs)
        infos.append(info)
        
        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            obs = eval_env.reset()  # Reset the environment for a new episode

    eval_env.close()
    return rewards, infos

# Evaluate the model
rewards, infos = evaluate_PPO_model(n_case=14, model=model)

print(rewards)

# Plot rewards over time
plt.plot(rewards, '-o')
plt.title('Reward at Each Time Step')
plt.xlabel('Time Steps')
plt.ylabel('Reward')
plt.show()

# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True, return_episode_rewards=False)
# reward_list, episode_len = evaluate_policy(model, eval_env, n_eval_episodes=1, callback=, deterministic=True, return_episode_rewards=True)

# print(f"Mean reward: {mean_reward} +/- {std_reward}")
# print(f"{episode_len} hours of Reward: {reward_list}")
