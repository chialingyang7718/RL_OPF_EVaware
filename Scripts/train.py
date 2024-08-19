#Dependency: torch, gym, stable_baselines3, pandapower, tensorflow

#Import the gym
import gymnasium as gym

#Import the registered environment
import CustEnv

#Import the grid loader
from grid_loader import load_test_case_grid

#Import the callback
from callback_soc import SOCCallback

#Import stable baselines stuff
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

#Import other stuff
import torch as th
import os
import re
import math


# initialize the vectorized environment
def make_env(env_id, net, dispatching_interval, EVaware, rank):
    def _init():
        env = gym.make(env_id, net=net, dispatching_intervals=dispatching_interval, EVaware=EVaware)
        env.reset(seed = rank)
        env = Monitor(env)
        return env
    return _init


if __name__ == '__main__':

    # Load the grid
    n_case = 14
    grid = load_test_case_grid(n_case)

    # Parameters
    env_id = 'PowerGrid-v0'
    num_envs = 6
    EV_aware = True

    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, 
                                  net=grid, 
                                  dispatching_interval=24,
                                  EVaware=EV_aware,
                                  rank=i) for i in range(num_envs)])


    # create the log path
    log_path = os.path.join('Training', 'Logs','10x')

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
    
    def create_unique_soc_log_path(base_log_dir):
        # Ensure the base log directory exists
        os.makedirs(base_log_dir, exist_ok=True)
        
        # List all items in the base log directory
        existing_items = os.listdir(base_log_dir)
        
        # Filter items that match the SOC naming pattern and extract numbers
        soc_numbers = [int(re.search(r'SOC(\d+)', item).group(1)) for item in existing_items if re.match(r'SOC\d+', item)]
        
        # Determine the next SOC number (start from 1 if no existing SOC directories)
        next_soc_number = max(soc_numbers) + 1 if soc_numbers else 1
        
        # Create the new SOC directory name
        new_soc_dir_name = f"SOC{next_soc_number}"
        new_soc_dir_path = os.path.join(base_log_dir, new_soc_dir_name)
        
        # Create the new SOC directory
        os.makedirs(new_soc_dir_path, exist_ok=True)
        
        return new_soc_dir_path


    # initialize callback
    soc_log_path = create_unique_soc_log_path(os.path.join(log_path, 'SOC'))
    soc_callback = SOCCallback(log_dir=soc_log_path)


    # create the agent
    model = PPO(policy="MlpPolicy", 
                env = env, 
                n_steps = 120, # update every day
                verbose = 0, 
                policy_kwargs = policy_kwargs, 
                tensorboard_log=log_path)
    


    # train the agent
    if EV_aware:
        model.learn(total_timesteps= 1000000, callback=[soc_callback], progress_bar=True)
    # model.learn(total_timesteps= 96, callback=[soc_callback], progress_bar=True)

    # save the model
    model.save("Training/Model/Case%s_EV" % n_case)

 


# DONE: why the model does not end -- Ans: the n_steps was much larger than total_timesteps
# DONE: implement time series data for simple case
# DONE: omit generator
# DONE: implement simbench data -- changed the network architecture
# DONE: include generation uncertainties into the observation -- added action limit at the step function
# DONE: vectorize the environment -- register the env and make vectorized env
# DONE: omit ext grid -- cannot omit it otherwise no reference bus is available
# DONE: integrate EV charging: address negative power in storage -- need to add penalty for negative SOC
# TODO: manually defined limits need to be adapted to the grid size e.g. q_g_mvar
# TODO: implement safe reinforcement learning
# DONE: generate instances (loads and max. power generations) of the enironment - randomly generate the loads and max. power generations
# TODO: output the report of divergence to the log fileS
# TODO: choose a comparable benchmark: interior point method, etc.
