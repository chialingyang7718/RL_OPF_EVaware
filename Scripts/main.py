#Dependency: torch, gym, stable_baselines3, pandapower, tensorflow

#Import the gym
import gymnasium as gym

#Import the registered environment
import CustEnv


#Import the grid loader
from grid_loader import load_test_case_grid, load_simple_grid, load_simbench_grid

#Import stable baselines stuff
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

#Import other stuff
import torch as th
import os


# initialize the vectorized environment
def make_env(env_id, net, dispatching_interval, UseDataSource, UseSimbench, rank):
    def _init():
        env = gym.make(env_id, net=net, dispatching_intervals=dispatching_interval, UseDataSource=UseDataSource, UseSimbench=UseSimbench)
        env.reset(seed = rank)
        env = Monitor(env)
        return env
    return _init


if __name__ == '__main__':

    # Load the grid
    # grid = load_test_case_grid(14)
    # grid = load_simple_grid()
    grid_code = "1-HV-urban--0-sw"
    grid = load_simbench_grid(grid_code)

    # Parameters
    env_id = 'PowerGrid-v0'
    num_envs = 4

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, 
                                  net=grid, 
                                  dispatching_interval=96, 
                                  UseDataSource=True, 
                                  UseSimbench=True, 
                                  rank=i) for i in range(num_envs)])


    # create the log path
    log_path = os.path.join('Training', 'Logs')

    # custom MLP policy: change network architecture into two layers of size 256
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
    model.learn(total_timesteps= 600000, progress_bar=True)

    # save the model
    # model.save("Training/Model/%s" % grid_code)


# DONE: why the model does not end -- Ans: the n_steps was much larger than total_timesteps
# DONE: implement time series data for simple case
# DONE: omit generator
# DONE: implement simbench data -- changed the network architecture
# DONE: include generation uncertainties into the observation -- added action limit at the step function
# DONE: vectorize the environment -- register the env and make vectorized env
# TODO: change the action into relative action??
# TODO: output the report of divergence to the log file
# TODO: integrate EV charging
# TODO: choose a comparable benchmark
