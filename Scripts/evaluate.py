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
import os
import seaborn as sns
import time
import pickle


def evaluate_PPO_model(n_steps=24, n_case=14, model=None,random_seed=None):
    # Load the environment
    grid = load_test_case_grid(n_case)
    # eval_env = gym.make('PowerGrid-v0', net=grid, dispatching_intervals=24, EVaware=True, Training=True)
    eval_env = gym.make(
        "PowerGrid-v2", net=grid, dispatching_intervals=n_steps, Training=False
    )

    # Initialize variables for tracking metrics
    truncated = False
    terminated = False

    # Evaluate the model
    obs, info = eval_env.reset()

    # Record EV battery capacity
    df_EV_spec = info["EV_spec"]

    # Initialize time list for RL model prediction
    RLTime = 0
    generation_cost = 0
    rewards = []
    bus_violation = 0
    line_violation = 0
    angle_violation = 0
    SOCviolation = 0

    for step in range(n_steps):
        # set start time for RL model prediction
        start = time.process_time()

        # Predict the action
        action, _state = model.predict(obs, deterministic=True)

        # Record the time taken for RL model prediction
        RLTime += time.process_time() - start

        obs, reward, terminated, truncated, info = eval_env.step(action)


        # Log the metrics
        # rewards
        rewards.append(round(reward, 3))

        # infos
        if step == 0:
            # Load
            df_load_p = pd.DataFrame(info["load_p"]).transpose()
            df_load_q = pd.DataFrame(info["load_q"]).transpose()

            # Renewables
            df_renewable = pd.DataFrame(info["renewable_p"]).transpose()

            # EV demand, SOC
            df_ev_demand = pd.DataFrame(info["EV_demand"]).transpose()
            df_ev_soc = pd.DataFrame(info["EV_SOC"]).transpose()
            # df_soc_threshold = pd.DataFrame(info["SOC_threshold"]).transpose()

            # Generation
            df_gen_p = pd.DataFrame(info["generation_p"]).transpose()
            df_gen_v = pd.DataFrame(info["generation_v"]).transpose()

            # EV charging
            df_ev_action = pd.DataFrame(info["EV_p"]).transpose()

            # Voltage
            df_voltage = pd.DataFrame(info["bus_voltage"]).transpose()

            # Line loading
            df_line_loading = pd.DataFrame(info["line_loading"]).transpose()

            # Bus Violation
            df_bus_violation = pd.DataFrame(info["bus_violation"]).transpose()

            # Line Violation
            df_line_violation = pd.DataFrame(info["line_violation"]).transpose()

            # Phase Angle Violation
            df_phase_angle_violation = pd.DataFrame(info["phase_angle_violation"]).transpose()
        else:
            ## State
            # Load
            df_load_p = pd.concat([df_load_p, pd.DataFrame(info["load_p"]).transpose()])
            df_load_q = pd.concat([df_load_q, pd.DataFrame(info["load_q"]).transpose()])

            # Renewables
            df_renewable = pd.concat(
                [df_renewable, pd.DataFrame(info["renewable_p"]).transpose()]
            )

            # EV demand, SOC
            df_ev_demand = pd.concat(
                [df_ev_demand, pd.DataFrame(info["EV_demand"]).transpose()]
            )
            df_ev_soc = pd.concat(
                [df_ev_soc, pd.DataFrame(info["EV_SOC"]).transpose()]
            )
            # df_soc_threshold = pd.concat(
            #     [df_soc_threshold, pd.DataFrame(info["SOC_threshold"]).transpose()]
            # )

            ## Action
            # Generation
            df_gen_p = pd.concat(
                [df_gen_p, pd.DataFrame(info["generation_p"]).transpose()]
            )
            # df_gen_q = pd.concat([df_gen_q, pd.DataFrame(info["generation_q"]).transpose()])
            df_gen_v = pd.concat(
                [df_gen_v, pd.DataFrame(info["generation_v"]).transpose()]
            )

            # EV charging
            df_ev_action = pd.concat(
                [df_ev_action, pd.DataFrame(info["EV_p"]).transpose()]
            )

            ## Others
            # Voltage
            df_voltage = pd.concat(
                [df_voltage, pd.DataFrame(info["bus_voltage"]).transpose()]
            )

            # Line loading
            df_line_loading = pd.concat(
                [df_line_loading, pd.DataFrame(info["line_loading"]).transpose()]
            )
     
            # Bus Violation
            # if 'bus_violation' in info:
            df_bus_violation = pd.concat(
                [df_bus_violation, pd.DataFrame(info["bus_violation"]).transpose()]
            )

            # Line Violation
            df_line_violation = pd.concat([df_line_violation,pd.DataFrame(info["line_violation"]).transpose()])


            # Phase Angle Violation
            df_phase_angle_violation = pd.concat([df_phase_angle_violation,pd.DataFrame(info["phase_angle_violation"]).transpose()])

        # Record the number of violations
        if info["bus_violation"].size != 0:
            bus_violation += 1
        
        if info["line_violation"].size != 0:
            line_violation += 1
        
        if len(info["phase_angle_violation"]) != 0:
            angle_violation += 1

        if info["SOC_violation"] != 0:
            SOCviolation += 1
    

        # Generation Cost
        generation_cost += info["generation_cost"]

        if terminated or truncated:
            # print(f"Episode finished after {step + 1} steps")
            # Set dataframes index
            df_load_p.reset_index(drop=True, inplace=True)
            df_load_q.reset_index(drop=True, inplace=True)
            df_renewable.reset_index(drop=True, inplace=True)
            df_ev_demand.reset_index(drop=True, inplace=True)
            df_ev_soc.reset_index(drop=True, inplace=True)
            # df_soc_threshold.reset_index(drop=True, inplace=True)
            df_gen_p.reset_index(drop=True, inplace=True)
            df_gen_v.reset_index(drop=True, inplace=True)
            df_ev_action.reset_index(drop=True, inplace=True)
            df_voltage.reset_index(drop=True, inplace=True)
            df_line_loading.reset_index(drop=True, inplace=True)
            df_bus_violation.reset_index(drop=True, inplace=True)
            df_line_violation.reset_index(drop=True, inplace=True)
            df_phase_angle_violation.reset_index(drop=True, inplace=True)
            # df_generation_cost = pd.DataFrame(generation_cost)


            """Save the dataframes to a csv file"""
            # Save the dataframes to a csv file
            save_all_df_to_csv(random_seed, n_case, df_EV_spec, df_load_p, df_load_q, df_renewable, df_ev_demand, df_ev_soc, df_gen_p, df_gen_v, df_ev_action, df_voltage, df_line_loading, df_bus_violation, df_line_violation, df_phase_angle_violation) #, df_generation_cost)
            

            # Reset the environment for a new episode
            obs, info = eval_env.reset()
    eval_env.close()

    return rewards, RLTime, generation_cost, bus_violation, line_violation, angle_violation, SOCviolation

def save_all_df_to_csv(random_seed, n_case, df_EV_spec, df_load_p, df_load_q, df_renewable, df_ev_demand, df_ev_soc, df_gen_p, df_gen_v, df_ev_action, df_voltage, df_line_loading, df_bus_violation, df_line_violation, df_phase_angle_violation):
    # Create the directory if it does not exist
        if not os.path.exists("Evaluation/Case%s/%s" %(n_case,random_seed)):
            os.makedirs("Evaluation/Case%s/%s" %(n_case,random_seed))

        # Save the dataframes to a csv file
        df_EV_spec.to_csv("Evaluation/Case%s/%s/EV_spec.csv" %(n_case,random_seed), index=False)
        # df_soc_threshold.to_csv("Evaluation/Case%s/soc_threshold.csv" %n_case, index=False)
        df_load_p.to_csv("Evaluation/Case%s/%s/load_p.csv" %(n_case,random_seed), index=False)
        df_load_q.to_csv("Evaluation/Case%s/%s/load_q.csv" %(n_case,random_seed), index=False)
        df_renewable.to_csv("Evaluation/Case%s/%s/renewable.csv" %(n_case,random_seed), index=False)
        df_ev_demand.to_csv("Evaluation/Case%s/%s/ev_demand.csv" %(n_case,random_seed), index=False)
        df_ev_soc.to_csv("Evaluation/Case%s/%s/ev_soc.csv" %(n_case,random_seed), index=False)
        df_gen_p.to_csv("Evaluation/Case%s/%s/gen_p.csv" %(n_case,random_seed), index=False)
        df_gen_v.to_csv("Evaluation/Case%s/%s/gen_v.csv" %(n_case,random_seed), index=False)
        df_ev_action.to_csv("Evaluation/Case%s/%s/ev_action.csv" %(n_case,random_seed), index=False)
        df_voltage.to_csv("Evaluation/Case%s/%s/voltage.csv" %(n_case,random_seed), index=False)
        df_line_loading.to_csv("Evaluation/Case%s/%s/line_loading.csv" %(n_case,random_seed), index=False)
        df_bus_violation.to_csv(
            "Evaluation/Case%s/%s/bus_violation.csv" %(n_case,random_seed), index=False
        )
        df_line_violation.to_csv(
            "Evaluation/Case%s/%s/line_violation.csv" %(n_case,random_seed), index=False
        )
        df_phase_angle_violation.to_csv(
            "Evaluation/Case%s/%s/phase_angle_violation.csv" %(n_case,random_seed), index=False
        )
        # df_generation_cost.to_csv(
        #     "Evaluation/Case%s/Case%s_%s/generation_cost.csv" %(n_case, n_case, EVScenario), index=False
        # )

if __name__ == "__main__":

    # EVScenarios = ["ImmediateFull", "ImmediateBalanced", "Home", "Night"]
    n_case = 30
    # n_case = int(input("Enter Test Case Number (9, 14, 30, 39, 57): "))
    starting_sample = 0
    sample_size = 1
    # sample_size = int(input('Enter the number of samples: '))
    # using dictionary comprehension to construct
    # Times = {EVScenario: [] for EVScenario in EVScenarios}
    # Costs = {EVScenario: [] for EVScenario in EVScenarios}
    # N_violations = {EVScenario: [] for EVScenario in EVScenarios}
    # N_violations_relax = {EVScenario: [] for EVScenario in EVScenarios}
    Times = []
    Costs = []
    N_bus_violations = []
    N_line_violations = []
    N_angle_violations = []
    N_SOC_violations = []


    for random_seed in range(starting_sample,sample_size+starting_sample):
        # for i in range(len(EVScenarios)):
        # for i in [1]:
        #     # Define Parameters
        #     EVScenario = EVScenarios[i]
        n_steps = 24
        # Load the trained model
        model = PPO.load("Training/Model/Case%s/Case%s" %(n_case, n_case))
        # model = PPO.load("Training/Model/Case%s/Case%s_2" %(n_case, n_case))
        model.set_random_seed(random_seed)
        # Evaluate the model
        rewards, Time, Cost, bus_violation, line_violation, angle_violation, SOCviolation = evaluate_PPO_model(
            n_steps=n_steps, n_case=n_case, model=model, random_seed=random_seed
        )
        
        # append the metrics to the lists
        Times.append(Time)
        Costs.append(Cost)
        N_bus_violations.append(bus_violation)
        N_line_violations.append(line_violation)
        N_angle_violations.append(angle_violation)
        N_SOC_violations.append(SOCviolation)
    
        print("Sample: ", random_seed)
        print("Total Rewards: ", sum(rewards))
        print("Generation Cost: ", Cost)
        print("Bus Violations: ", bus_violation, "Line Violations: ", line_violation, "Angle Violations: ", angle_violation, "SOC Violations: ", SOCviolation, "\n")



    # Save the metrics dicts
    with open(f"Evaluation/Case{n_case}/{random_seed}/RL_metrics.pkl" , "wb") as f:
        pickle.dump([Time, Cost, N_bus_violations, N_line_violations, N_angle_violations, N_SOC_violations], f)


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
