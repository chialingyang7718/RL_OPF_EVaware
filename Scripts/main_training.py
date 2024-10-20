# Dependency: torch, gym, stable_baselines3, pandapower, tensorflow

# Import the gym
import gymnasium as gym

# Import the registered environment
import CustEnv

# Import the grid loader
from grid_loader import load_test_case_grid

# Import the callback
# from callback_soc import SOCCallback

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
def make_env(env_id, net, dispatching_interval, EVScenario, Training, rank):
    def _init():
        env = gym.make(
            env_id,
            net=net,
            dispatching_intervals=dispatching_interval,
            EVScenario=EVScenario,
            Training=Training,
        )
        env.reset(seed=rank)
        env = Monitor(env)
        return env

    return _init

# find the nearest power of 2 to the number of buses
def nearestPowerOf2(N):
    a = int(math.log2(N))
    if 2**a == N:
        return N
    return 2 ** (a + 1)  # return the ceiling of the power of 2

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
    n_case = 30
    grid = load_test_case_grid(n_case)

    # Parameters
    env_id = "PowerGrid-v1"
    num_envs = 6
    Training = True
    EVScenarios = ["ImmediateFull", "ImmediateBalanced", "Home", "Night"]
    
    for i in [1]:
    # for i in [0, 1, 2, 3]:
        EVScenario = EVScenarios[i]

        # Create the vectorized environment
        env = SubprocVecEnv(
            [
                make_env(
                    env_id,
                    net=grid,
                    dispatching_interval=24,
                    EVScenario=EVScenario,
                    Training=Training,
                    rank=i,
                )
                for i in range(num_envs)
            ]
        )

        # create the log path
        log_path = os.path.join("Training", "Logs", "Case%s"% n_case, EVScenario)

        # custom MLP policy: network depends on the observation space, indirectly, the number of buses


        # choose a network size that is slightly larger than observation space
        NN_size = nearestPowerOf2(n_case * 4)  
        

        # the policy network architecture (if the number of buses is less than 20, 2 layers; otherwise, 3 layers)
        if n_case <= 20:
            n_steps = 120
            total_timesteps = 100000 * (int(n_case/10) + 3)
        elif n_case > 20:
            n_steps = 60
            total_timesteps = 100000 * (int(n_case/10) + 5)
        
        # n_layers = 2 + int(n_case/10)
        policy_kwargs = dict(
                activation_fn=th.nn.Tanh,
                net_arch=dict(pi=[NN_size*4, NN_size*2, NN_size*2, NN_size, NN_size], vf=[NN_size*4, int(NN_size/2), int(NN_size/4)])  # at max 3 layers for value function
            )


        # initialize callback
        # soc_log_path = create_unique_soc_log_path(os.path.join(log_path, "SOC"))
        # soc_callback = SOCCallback(log_dir=soc_log_path)

        # create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=os.path.join(log_path, "Checkpoints"),
            name_prefix="Case%s_model"%n_case,
            )

        # create the agent or reload the model
        if os.path.exists("Training/Model/Case%s/Case%s_%s.zip"%(n_case, n_case, EVScenario)):    # check if the model exists:
            model = PPO.load("Training/Model/Case%s/Case%s_%s"%(n_case, n_case, EVScenario))
            total_timesteps = 600000
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

        # train the agent
        if EVScenario is not None:
            # model.learn(total_timesteps=60, progress_bar=True)
            model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=True)

        # save the model
        model.save("Training/Model/Case%s/Case%s_"%(n_case, n_case) + EVScenario)
        env.close()

# DONE: why the model does not end -- Ans: the n_steps was much larger than total_timesteps
# DONE: implement time series data for simple case
# DONE: omit generator
# DONE: implement simbench data -- changed the network architecture
# DONE: include generation uncertainties into the observation -- added action limit at the step function
# DONE: vectorize the environment -- register the env and make vectorized env
# DONE: omit ext grid -- cannot omit it otherwise no reference bus is available
# DONE: integrate EV charging: address negative power in storage -- need to add penalty for negative SOC
# DONE: generate instances (loads and max. power generations) of the enironment - randomly generate the loads and max. power generations
# DONE: exclude slack bus from generation cost since the slack bus may represent a large external grid or network that is assumed to have sufficient capacity without a direct cost tied to its power output
# DONE: seperate the vlimit of generator and other buses

# DONE: add phase angle difference limit into reward function
# DONE: read voltage limit from the grid
# TODO: output the report of divergence to the log fileS
# TODO: choose a comparable benchmark: interior point method, etc.
# TODO: implement safe reinforcement learning
