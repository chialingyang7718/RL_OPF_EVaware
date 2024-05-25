#Dependency: torch, gym, stable_baselines3, pandapower, tensorflow

#Import the gym
import gymnasium as gym

#Import the registered environment
import CustEnv
# from environment.PowerGridEnv import PowerGrid

#Import the grid loader
from grid_loader import load_test_case_grid, load_simple_grid, load_simbench_grid

#Import stable baselines stuff
from stable_baselines3 import PPO

#Import other stuff
import torch as th
import os



# Environment Setup
# grid = load_test_case_grid(14)
# grid = load_simple_grid()
grid_code = "1-HV-urban--0-sw"
grid = load_simbench_grid(grid_code)
# env = PowerGrid(grid, 0.03, 96, True, True) 
env = gym.make('PowerGrid-v0', net = grid, dispatching_intervals = 96, num_envs = 2)


# create the log path
log_path = os.path.join('Training', 'Logs')

# custom MLP policy of two layers of size 256
policy_kwargs = dict(
    activation_fn=th.nn.Tanh, net_arch=dict(pi=[256, 256], vf=[256, 256])
)

# create the agent
model = PPO(policy="MlpPolicy", 
            env = env, 
            n_steps = 100, 
            verbose=1, 
            policy_kwargs = policy_kwargs, 
            tensorboard_log=log_path)

# train the agent
model.learn(total_timesteps= 60000, progress_bar=True)

# save the model
model.save("Training/Model/%s" % grid_code)


# DONE: why the model does not end Ans: the n_steps was much larger than total_timesteps
# DONE: implement time series data for simple case
# DONE: split the script into multiple files
# DONE: omit generator
# DONE: implement simbench data -- change the network architecture
# DONE: include generation uncertainties into the observation - add action limit at the step function
# TODO: vectorize the environment: register the env and make vectorized env
# TODO: change the action into relative action??
# TODO: Output the report of divergence to the log file
# TODO: integrate EV charging
