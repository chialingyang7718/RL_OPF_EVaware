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
import math


# initialize the vectorized environment
def make_env(env_id, net, dispatching_interval, UseDataSource, UseSimbench, EVaware, rank):
    def _init():
        env = gym.make(env_id, net=net, dispatching_intervals=dispatching_interval, UseDataSource=UseDataSource, UseSimbench=UseSimbench, EVaware=EVaware)
        env.reset(seed = rank)
        env = Monitor(env)
        return env
    return _init


if __name__ == '__main__':

    # Load the grid
    n_case = 14
    grid = load_test_case_grid(n_case)
    # grid = load_simple_grid()
    # grid_code = "1-HV-urban--0-sw"
    # grid = load_simbench_grid(grid_code)

    # Parameters
    env_id = 'PowerGrid-v0'
    num_envs = 4

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, 
                                  net=grid, 
                                  dispatching_interval=24, 
                                  UseDataSource=False, 
                                  UseSimbench=False,
                                  EVaware=True,
                                  rank=i) for i in range(num_envs)])


    # create the log path
    # log_path = os.path.join('Training', 'Logs')

    # custom MLP policy: network depends on the observation space, indirectly, the number of buses
    # first, we need to find the nearest power of 2 to the number of buses
    def nearestPowerOf2(N):
        a = int(math.log2(N))
        if 2**a == N:
            return N
        return 2**(a + 1) # return the ceiling of the power of 2
    
    # choose a network size that is slightly larger than the observation space
    NN_size = nearestPowerOf2(n_case * 4) # 4 times of the number of buses is slightly larger the observation space which contains sgen_pmw, load_pmw, and load_qmvar & SOC
    
    # the policy network architecture
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh, net_arch=dict(pi=[NN_size, NN_size], vf=[NN_size, NN_size])
    )

    # create the agent
    model = PPO(policy="MlpPolicy", 
                env = env, 
                n_steps = 12, 
                verbose=1, 
                policy_kwargs = policy_kwargs)#, 
                # tensorboard_log=log_path)


    # train the agent
    model.learn(total_timesteps= 96, progress_bar=True)

    # save the model
    # model.save("Training/Model/%s" % grid_code)


# DONE: why the model does not end -- Ans: the n_steps was much larger than total_timesteps
# DONE: implement time series data for simple case
# DONE: omit generator
# DONE: implement simbench data -- changed the network architecture
# DONE: include generation uncertainties into the observation -- added action limit at the step function
# DONE: vectorize the environment -- register the env and make vectorized env
# DONE: omit ext grid -- cannot omit it otherwise no reference bus is available
# TODO: integrate EV charging: discharging efficiency, address negative power in storage
# TODO: manually defined limits need to be adapted to the grid size e.g. q_g_mvar
# TODO: implement safe reinforcement learning
# TODO: generate instances (loads and max. power generations) of the enironment
# TODO: change the action into relative action??
# TODO: output the report of divergence to the log fileS
# TODO: choose a comparable benchmark: interior point method, etc.
