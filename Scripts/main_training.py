# Dependency: torch, gym, stable_baselines3, pandapower, tensorflow

# Import the gym
import gymnasium as gym

# Import the registered environment
import CustEnv

# Import the grid loader
from grid_loader import load_test_case_grid

# Import stable baselines stuff
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Import other stuff
import torch as th
import os
import re
import math


# initialize the vectorized environment
def make_env(env_id, net, dispatching_interval, Training, rank):
    def _init():
        env = gym.make(
            env_id,
            net=net,
            dispatching_intervals=dispatching_interval,
            Training=Training,
        )
        env.reset(seed=rank)
        env = Monitor(env)
        return env
    return _init

def nearestPowerOf2(N):
    a = int(math.log2(N))
    if 2**a == N:
        return N
    return 2 ** (a + 1)

def create_unique_soc_log_path(base_log_dir):
        # Ensure the base log directory exists
        os.makedirs(base_log_dir, exist_ok=True)

        # List all items in the base log directory
        existing_items = os.listdir(base_log_dir)

        # Filter items that match the SOC naming pattern and extract numbers
        soc_numbers = [
            int(re.search(r"SOC(\d+)", item).group(1))
            for item in existing_items
            if re.match(r"SOC\d+", item)
        ]

        # Determine the next SOC number (start from 1 if no existing SOC directories)
        next_soc_number = max(soc_numbers) + 1 if soc_numbers else 1

        # Create the new SOC directory name
        new_soc_dir_name = f"SOC{next_soc_number}"
        new_soc_dir_path = os.path.join(base_log_dir, new_soc_dir_name)

        # Create the new SOC directory
        os.makedirs(new_soc_dir_path, exist_ok=True)

        return new_soc_dir_path

if __name__ == "__main__":

    # Load the grid
    n_case = 5
    grid = load_test_case_grid(n_case)

    # Parameters
    env_id = "PowerGrid-v2"
    num_envs = 6
    Training = True

    # Create the vectorized environment
    env = SubprocVecEnv(
        [
            make_env(
                env_id,
                net=grid,
                dispatching_interval=24,
                Training=Training,
                rank=i,
            )
            for i in range(num_envs)
        ]
    )

    # create the log path
    log_path = os.path.join("Training", "Logs", "Case%s" % (n_case))

    NN_size = nearestPowerOf2(n_case * 4) 
    n_steps = 120

    # choose network architecture depending on the case number
    if n_case <= 20:
        n_layers = 2 + int(n_case/10)
        policy_kwargs = dict(
                activation_fn=th.nn.Tanh,
                net_arch=dict(pi=[NN_size] * n_layers, vf=[NN_size] * min(n_layers, 3))  # at max 3 layers for value function
            )

    elif n_case > 20:
        if n_case == 30 or n_case == 33:
            policy_kwargs = dict(
                    activation_fn=th.nn.Tanh,
                    net_arch=dict(pi=[NN_size*4, NN_size*2, NN_size*2, NN_size, NN_size], vf=[NN_size*4, NN_size, NN_size])  
                )
        elif n_case == 39:
            policy_kwargs = dict(
                    activation_fn=th.nn.Tanh,
                    net_arch=dict(pi=[NN_size*4, NN_size*2, NN_size*2, NN_size*2, NN_size, NN_size], vf=[NN_size*4, NN_size*2, NN_size/2])  
                )
        elif n_case == 57:
            policy_kwargs = dict(
                    activation_fn=th.nn.Tanh,
                    net_arch=dict(pi=[NN_size*4, NN_size*2, NN_size*2, NN_size*2, NN_size, NN_size], vf=[NN_size*4, int(NN_size/2), int(NN_size/4)]) 
                )
        
    # create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=os.path.join(log_path, "Checkpoints"),
        name_prefix="Case%s_model"%n_case,
        )

    # continue training or start new training
    if os.path.exists("Training/Model/Case%s/Case%s.zip" %(n_case, n_case)):   
        model = PPO.load("Training/Model/Case%s/Case%s" %(n_case, n_case), env=env)
        model.learn(total_timesteps=200000, reset_num_timesteps=False, tb_log_name="PPO_continue", callback=checkpoint_callback, progress_bar=True)

    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=n_steps,
            batch_size= int(num_envs* n_steps/10),
            gamma=0.99,
            verbose=0,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_path,
        )
        total_timesteps = 500000
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=True)

    model.save("Training/Model/Case%s/Case%s" %(n_case, n_case))
    env.close()

