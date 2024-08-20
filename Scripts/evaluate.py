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
import pandas as pd
import json
import os


def evaluate_PPO_model(n_case=14, model=None):
    # Load the environment
    grid = load_test_case_grid(n_case)
    eval_env = gym.make('PowerGrid-v0', net=grid, dispatching_intervals=24, EVaware=True)

    # Initialize variables for tracking metrics
    n_steps = 24  
    rewards = []
    truncated = False
    terminated = False
    # infos = {}


    # Evaluate the model
    obs, info = eval_env.reset()
    for step in range(n_steps):
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        
        # Log the metrics
        # rewards
        rewards.append(round(reward,3))
        # infos
        if step == 0:
            ### Convert the infos to a pandas dataframe
            ## State
            # Load
            df_load_p = pd.DataFrame(info["load_p"]).transpose()
            df_load_q = pd.DataFrame(info["load_q"]).transpose()

            # Renewables
            df_renewable = pd.DataFrame(info["renewable_p"]).transpose()

            # EV demand, SOC
            df_ev_demand = pd.DataFrame(info["EV_demand"]).transpose()
            df_ev_soc = pd.DataFrame(info["EV_SOC_beginning"]).transpose()

            ## Action
            # Generation
            df_gen_p = pd.DataFrame(info["generation_p"]).transpose()
            df_gen_q = pd.DataFrame(info["generation_q"]).transpose()

            # EV charging
            df_ev_action = pd.DataFrame(info["EV_p"]).transpose()

            ## Others
            # Voltage
            df_voltage = pd.DataFrame(info["bus_voltage"]).transpose()

            # Line loading
            df_line_loading = pd.DataFrame(info["line_loading"]).transpose()

        else:
            ## State
            # Load
            df_load_p = pd.concat([df_load_p, pd.DataFrame(info["load_p"]).transpose()])
            df_load_q = pd.concat([df_load_q, pd.DataFrame(info["load_q"]).transpose()])


            # Renewables
            df_renewable = pd.concat([df_renewable, pd.DataFrame(info["renewable_p"]).transpose()])


            # EV demand, SOC
            df_ev_demand = pd.concat([df_ev_demand, pd.DataFrame(info["EV_demand"]).transpose()])
            df_ev_soc = pd.concat([df_ev_soc, pd.DataFrame(info["EV_SOC_beginning"]).transpose()])

            ## Action
            # Generation
            df_gen_p = pd.concat([df_gen_p, pd.DataFrame(info["generation_p"]).transpose()])
            df_gen_q = pd.concat([df_gen_q, pd.DataFrame(info["generation_q"]).transpose()])

            # EV charging
            df_ev_action = pd.concat([df_ev_action, pd.DataFrame(info["EV_p"]).transpose()])
   
            ## Others
            # Voltage
            df_voltage = pd.concat([df_voltage, pd.DataFrame(info["bus_voltage"]).transpose()])

            # Line loading
            df_line_loading = pd.concat([df_line_loading, pd.DataFrame(info["line_loading"]).transpose()])
        

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")

            # Set dataframes index
            df_load_p.reset_index(drop=True, inplace=True)
            df_load_q.reset_index(drop=True, inplace=True)
            df_renewable.reset_index(drop=True, inplace=True)
            df_ev_demand.reset_index(drop=True, inplace=True)
            df_ev_soc.reset_index(drop=True, inplace=True)
            df_gen_p.reset_index(drop=True, inplace=True)
            df_gen_q.reset_index(drop=True, inplace=True)
            df_ev_action.reset_index(drop=True, inplace=True)
            df_voltage.reset_index(drop=True, inplace=True)
            df_line_loading.reset_index(drop=True, inplace=True)


            # Create the directory if it does not exist
            if not os.path.exists("Evaluation/Case14_EV"):
                os.makedirs("Evaluation/Case14_EV")

            # Save the dataframes to a csv file
            df_load_p.to_csv("Evaluation/Case14_EV/load_p.csv", index=False)
            df_load_q.to_csv("Evaluation/Case14_EV/load_q.csv", index=False)
            df_renewable.to_csv("Evaluation/Case14_EV/renewable.csv", index=False)
            df_ev_demand.to_csv("Evaluation/Case14_EV/ev_demand.csv", index=False)
            df_ev_soc.to_csv("Evaluation/Case14_EV/ev_soc.csv", index=False)
            df_gen_p.to_csv("Evaluation/Case14_EV/gen_p.csv", index=False)
            df_gen_q.to_csv("Evaluation/Case14_EV/gen_q.csv", index=False)
            df_ev_action.to_csv("Evaluation/Case14_EV/ev_action.csv", index=False)
            df_voltage.to_csv("Evaluation/Case14_EV/voltage.csv", index=False)
            df_line_loading.to_csv("Evaluation/Case14_EV/line_loading.csv", index=False)

            # Reset the environment for a new episode
            obs, info = eval_env.reset()
    eval_env.close()
    return rewards, df_load_p, df_load_q, df_renewable, df_ev_demand, df_ev_soc, df_gen_p, df_gen_q, df_ev_action, df_voltage, df_line_loading

if __name__ == "__main__":
    # Load the trained model
    model = PPO.load("Training/Model/Case14_EV")

    # Evaluate the model
    rewards,  df_load_p, df_load_q, df_renewable, df_ev_demand, df_ev_soc, df_gen_p, df_gen_q, df_ev_action, df_voltage, df_line_loading = evaluate_PPO_model(n_case=14, model=model)


    # Plot rewards over time
    plt.plot(rewards, '-o')
    plt.title('Reward at Each Time Step')
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    for i in range(len(rewards)):                       
        plt.annotate('%s' % rewards[i], xy=(i,rewards[i]), textcoords='data')
    plt.show()
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True, return_episode_rewards=False)
    # reward_list, episode_len = evaluate_policy(model, eval_env, n_eval_episodes=1, callback=, deterministic=True, return_episode_rewards=True)

    # print(f"Mean reward: {mean_reward} +/- {std_reward}")
    # print(f"{episode_len} hours of Reward: {reward_list}")
