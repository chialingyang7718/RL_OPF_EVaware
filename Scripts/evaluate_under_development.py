# Import the registered environment
import CustEnv

# Import the grid loader
from grid_loader import load_test_case_grid

# Import the gym
import gymnasium as gym

# Import stable baselines stuff
from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy

# Import other stuff
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import time
import pickle


def evaluate_PPO_model(n_steps=24, n_case=14, EVScenario=None, model=None):
    # Load the environment
    grid = load_test_case_grid(n_case)
    
    eval_env = gym.make(
        "PowerGrid-v1", net=grid, dispatching_intervals=n_steps, EVScenario=EVScenario, Training=False
    )

    # Initialize variables for tracking metrics
    rewards = []
    generation_cost = []
    RLtime = []
    truncated = False
    terminated = False
    load_p = {}
    load_q = {}
    renewable_p = {}
    EV_demand = {}
    EV_SOC = {}
    generation_p = {}
    generation_v = {}
    EV_p = {}
    bus_voltage = {}
    line_loading = {}
    bus_violation = {}
    line_violation = {}
    phase_angle_violation = {}

 
    obs, info = eval_env.reset()
    # Record EV battery capacity
    df_EV_spec = info["EV_spec"]
   

    for step in range(n_steps):
        # set start time for RL model prediction
        start = time.process_time()

        # Predict the action
        action, _state = model.predict(obs, deterministic=False)
        
        # Record the time taken for RL model prediction
        RLtime.append(time.process_time() - start)
        
        # Update the state, reward and info...
        obs, reward, terminated, truncated, info = eval_env.step(action)

        # infos
        load_p, load_q, renewable_p, EV_demand, EV_SOC, generation_p, generation_v, EV_p, bus_voltage, line_loading, bus_violation, line_violation, phase_angle_violation\
        = convert_info_to_dict(step, info, load_p, load_q, renewable_p, EV_demand, EV_SOC, generation_p, generation_v, EV_p, bus_voltage, line_loading, bus_violation, line_violation, phase_angle_violation)
        
	    # Record cost, and reward
        rewards.append(round(reward, 3))
        generation_cost.append(info["generation_cost"])
        
        # SOC Violation
        if info["soc_violation"]:
            SOCviolation = 1
        else:
            SOCviolation = 0
                

        if terminated or truncated:
            df_load_p = pd.DataFrame(load_p).T
            df_load_q = pd.DataFrame(load_q).T
            df_renewable = pd.DataFrame(renewable_p).T
            df_ev_demand = pd.DataFrame(EV_demand).T
            df_ev_soc = pd.DataFrame(EV_SOC).T
            df_gen_p = pd.DataFrame(generation_p).T
            df_gen_v = pd.DataFrame(generation_v).T
            df_ev_action = pd.DataFrame(EV_p).T
            df_voltage = pd.DataFrame(bus_voltage).T
            df_line_loading = pd.DataFrame(line_loading).T
            df_bus_violation = pd.DataFrame(bus_violation).T
            df_line_violation = pd.DataFrame(line_violation).T
            df_phase_angle_violation = pd.DataFrame(phase_angle_violation).T

            print(df_load_p)

            # print(f"Episode finished after {step + 1} steps"
            save_all_df_to_csv(n_case, EVScenario, df_EV_spec, df_load_p, df_load_q, df_renewable, df_ev_demand, df_ev_soc, df_gen_p, df_gen_v, df_ev_action, df_voltage, df_line_loading, df_bus_violation, df_line_violation, df_phase_angle_violation)
            
            """Metrices"""
            # Record the time taken for RL model prediction
            Time = sum(RLtime)
            Cost = sum(generation_cost)

            # Record the violation

            if df_bus_violation.empty & df_line_violation.empty & df_phase_angle_violation.empty:
                N_violation_relax = 0
                N_violation = SOCviolation
            else:
                N_violation_relax = 1
                N_violation = 1

            # Reset the environment for a new episode
            obs, info = eval_env.reset()
    eval_env.close()
    return rewards, Time, Cost, N_violation, N_violation_relax

def convert_info_to_dict(step, info, load_p, load_q, renewable_p, EV_demand, EV_SOC, generation_p, generation_v, EV_p, bus_voltage, line_loading, bus_violation, line_violation, phase_angle_violation):
    load_p[step] = info["load_p"]
    load_q[step] = info["load_q"]
    renewable_p[step] = info["renewable_p"]
    EV_demand[step] = info["EV_demand"]
    EV_SOC[step] = info["EV_SOC_beginning"]
    generation_p[step] = info["generation_p"]
    generation_v[step] = info["generation_v"]
    EV_p[step] = info["EV_p"]
    bus_voltage[step] = info["bus_voltage"]
    line_loading[step] = info["line_loading"]
    bus_violation[step] = info["bus_violation"]
    line_violation[step] = info["line_violation"]
    phase_angle_violation[step] = info["phase_angle_violation"]
    return load_p, load_q, renewable_p, EV_demand, EV_SOC, generation_p, generation_v, EV_p, bus_voltage, line_loading, bus_violation, line_violation, phase_angle_violation

def save_all_df_to_csv(n_case, EVScenario, df_EV_spec, df_load_p, df_load_q, df_renewable, df_ev_demand, df_ev_soc, df_gen_p, df_gen_v, df_ev_action, df_voltage, df_line_loading, df_bus_violation, df_line_violation, df_phase_angle_violation, df_generation_cost):
    	# Create the directory if it does not exist
        if not os.path.exists("Evaluation/Case%s/Case%s_%s" %(n_case, n_case, EVScenario)):
            os.makedirs("Evaluation/Case%s/Case%s_%s" %(n_case, n_case, EVScenario))

        # Save the dataframes to a csv file
        df_EV_spec.to_csv("Evaluation/Case%s/Case%s_%s/EV_spec.csv" %(n_case, n_case, EVScenario), index=False)
        df_load_p.to_csv("Evaluation/Case%s/Case%s_%s/load_p.csv" %(n_case, n_case, EVScenario), index=False)
        df_load_q.to_csv("Evaluation/Case%s/Case%s_%s/load_q.csv" %(n_case, n_case, EVScenario), index=False)
        df_renewable.to_csv("Evaluation/Case%s/Case%s_%s/renewable.csv" %(n_case, n_case, EVScenario), index=False)
        df_ev_demand.to_csv("Evaluation/Case%s/Case%s_%s/ev_demand.csv" %(n_case, n_case, EVScenario), index=False)
        df_ev_soc.to_csv("Evaluation/Case%s/Case%s_%s/ev_soc.csv" %(n_case, n_case, EVScenario), index=False)
        df_gen_p.to_csv("Evaluation/Case%s/Case%s_%s/gen_p.csv" %(n_case, n_case, EVScenario), index=False)
        df_gen_v.to_csv("Evaluation/Case%s/Case%s_%s/gen_v.csv" %(n_case, n_case, EVScenario), index=False)
        df_ev_action.to_csv("Evaluation/Case%s/Case%s_%s/ev_action.csv" %(n_case, n_case, EVScenario), index=False)
        df_voltage.to_csv("Evaluation/Case%s/Case%s_%s/voltage.csv" %(n_case, n_case, EVScenario), index=False)
        df_line_loading.to_csv("Evaluation/Case%s/Case%s_%s/line_loading.csv" %(n_case, n_case, EVScenario), index=False)
        df_bus_violation.to_csv(
            "Evaluation/Case%s/Case%s_%s/bus_violation.csv" %(n_case, n_case, EVScenario), index=False
        )
        df_line_violation.to_csv(
            "Evaluation/Case%s/Case%s_%s/line_violation.csv" %(n_case, n_case, EVScenario), index=False
        )
        df_phase_angle_violation.to_csv(
            "Evaluation/Case%s/Case%s_%s/phase_angle_violation.csv" %(n_case, n_case, EVScenario), index=False
        )
        df_generation_cost.to_csv(
            "Evaluation/Case%s/Case%s_%s/generation_cost.csv" %(n_case, n_case, EVScenario), index=False
        )

if __name__ == "__main__":

    EVScenarios = ["ImmediateFull", "ImmediateBalanced", "Home", "Night"]
    n_case = 9
    # n_case = int(input("Enter Test Case Number (9, 14, 30, 39, 57): "))
    sample_size = 1
    # sample_size = int(input('Enter the number of samples: '))
    # using dictionary comprehension to construct
    Times = {EVScenario: [] for EVScenario in EVScenarios}
    Costs = {EVScenario: [] for EVScenario in EVScenarios}
    N_violations = {EVScenario: [] for EVScenario in EVScenarios}
    N_violations_relax = {EVScenario: [] for EVScenario in EVScenarios}

    for random_seed in range(sample_size):
        # for i in range(len(EVScenarios)):
        for i in range(1, 2):
            # Define Parameters
            EVScenario = EVScenarios[i]
            n_steps = 24
            # Load the trained model
            model = PPO.load("Training/Model/Case%s/PPO2/Case%s_%s" %(n_case, n_case, EVScenario))
            # model = PPO.load("Training/Model/Case%s/Case%s_2" %(n_case, n_case))
            model.set_random_seed(random_seed)
            # Evaluate the model
            rewards, Time, Cost, N_violation, N_violation_relax = evaluate_PPO_model(
                n_steps=n_steps, n_case=n_case, EVScenario=EVScenario, model=model
            )
            Times[EVScenario].append(Time)
            Costs[EVScenario].append(Cost)
            N_violations[EVScenario].append(N_violation)
            N_violations_relax[EVScenario].append(N_violation_relax)

            print(f"{EVScenario} Mean Reward: {sum(rewards)/n_steps}")


    # Save the metrics dicts
    with open("Evaluation/Case%s/Case%s_metrics.pkl" %(n_case, n_case), "wb") as f:
        pickle.dump([Times, Costs, N_violations, N_violations_relax], f)


        # # Plot rewards over time
        # plt.plot(rewards, "-o")
        # plt.title("Reward at Each Time Step")
        # plt.xlabel("Time Steps")
        # plt.ylabel("Reward")
        # for i in range(len(rewards)):
        #     plt.annotate("%s" % rewards[i], xy=(i, rewards[i]), textcoords="data")
        # plt.show()
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True, return_episode_rewards=False)
    # reward_list, episode_len = evaluate_policy(model, eval_env, n_eval_episodes=1, callback=, deterministic=True, return_episode_rewards=True)

    # print(f"Mean reward: {mean_reward} +/- {std_reward}")
