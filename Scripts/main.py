#Dependency: torch, gym, stable_baselines3, pandapower, tensorflow

#Import the gym
import gymnasium as gym

#Import the registered environment
import CustEnv


#Import the grid loader
from grid_loader import load_test_case_grid, load_simple_grid, load_simbench_grid

#Import stable baselines stuff
from stable_baselines3 import PPO
from stable_baselines3 import common

#Import other stuff
import torch as th
import os


# Load the grid
# grid = load_test_case_grid(14)
# grid = load_simple_grid()
grid_code = "1-HV-urban--0-sw"
grid = load_simbench_grid(grid_code)

# Environment Setup
# env = PowerGrid(grid, 0.03, 96, True, True) 
if __name__ == '__main__':
# initiate the environment for vectorizing
    env_fns = [
        lambda: gym.make(
            'PowerGrid-v0', 
            net = grid, 
            dispatching_intervals = 2, 
            UseDataSource = True, 
            UseSimbench = True
            )] * 2

    # vectorize the environment
    # env = gym.vector.AsyncVectorEnv(env_fns, shared_memory=True)
    env = common.vec_env.SubprocVecEnv(env_fns, start_method=None)

    # create the log path
    log_path = os.path.join('Training', 'Logs')

    # custom MLP policy of two layers of size 256
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh, net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    # create the agent
    model = PPO(policy="MlpPolicy", 
                env = env, 
                n_steps = 2, 
                verbose=1, 
                policy_kwargs = policy_kwargs, 
                tensorboard_log=log_path)

    # train the agent
    model.learn(total_timesteps= 60, progress_bar=True)

    # save the model
    # model.save("Training/Model/%s" % grid_code)


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
