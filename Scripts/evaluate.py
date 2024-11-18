# Import the registered environment
import CustEnv

# Import the grid loader
from grid_loader import load_test_case_grid

# Import the gym
import gymnasium as gym

# Import stable baselines stuff
from stable_baselines3 import PPO

# Import other stuff
import pandas as pd
import os
import time
import pickle


def evaluate_PPO_model(n_steps=24, n_case=14, model=None,random_seed=None, generate_csv=False):
    # Load the environment
    grid = load_test_case_grid(n_case)
    eval_env = gym.make(
        "PowerGrid", net=grid, dispatching_intervals=n_steps, Training=False
    )

    # Initialize variables for tracking metrics
    truncated = False
    terminated = False
    obs, info = eval_env.reset()

    df_EV_spec = info["EV_spec"]

    # Initialize time list for RL model prediction
<<<<<<< HEAD
    RLtime = []
    generation_cost = []
=======
    RLTime = 0
    generation_cost = 0
    rewards = []
    bus_violation = 0
    line_violation = 0
    angle_violation = 0
    SOCviolation = 0
>>>>>>> origin/env/omit_EVScenario

    for step in range(n_steps):
        start = time.process_time()
<<<<<<< HEAD

        # Predict the action
        action, _state = model.predict(obs, deterministic=False)
        # Record the time taken for RL model prediction
        RLtime.append(time.process_time() - start)
        obs, reward, terminated, truncated, info = eval_env.step(action)


=======
        action, _state = model.predict(obs, deterministic=True)
        RLTime += time.process_time() - start
        obs, reward, terminated, truncated, info = eval_env.step(action)

>>>>>>> origin/env/omit_EVScenario
        # Log the metrics
        rewards.append(round(reward, 3))
        if generate_csv:
            # infos
            if step == 0:
                # Load
                df_load_p = pd.DataFrame(info["load_p"]).transpose()
                df_load_q = pd.DataFrame(info["load_q"]).transpose()

<<<<<<< HEAD
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
 
            # SOC Violation
            if info["soc_violation"]:
                SOCviolation = 1
            else:
                SOCviolation = 0

            # Generation Cost
            generation_cost.append(info["generation_cost"])

        if terminated or truncated:
            # print(f"Episode finished after {step + 1} steps")

            # Set dataframes index
            df_load_p.reset_index(drop=True, inplace=True)
            df_load_q.reset_index(drop=True, inplace=True)
            df_renewable.reset_index(drop=True, inplace=True)
            df_ev_demand.reset_index(drop=True, inplace=True)
            df_ev_soc.reset_index(drop=True, inplace=True)
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
            save_all_df_to_csv(n_case, EVScenario, df_EV_spec, df_load_p, df_load_q, df_renewable, df_ev_demand, df_ev_soc, df_gen_p, df_gen_v, df_ev_action, df_voltage, df_line_loading, df_bus_violation, df_line_violation, df_phase_angle_violation) #, df_generation_cost)
            
            """Metrices"""
            # Record the time taken for RL model prediction
            Time = sum(RLtime)

            # Record the total cost
            Cost = sum(generation_cost)
=======
                # Renewables
                df_renewable = pd.DataFrame(info["renewable_p"]).transpose()

                # EV demand, SOC
                df_ev_demand = pd.DataFrame(info["EV_demand"]).transpose()
                df_ev_soc = pd.DataFrame(info["EV_SOC"]).transpose()
                # df_soc_threshold = pd.DataFrame(info["SOC_threshold"]).transpose()
>>>>>>> origin/env/omit_EVScenario

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
            
        generation_cost += info["generation_cost"]

        if generate_csv:
            if terminated or truncated:
                df_load_p.reset_index(drop=True, inplace=True)
                df_load_q.reset_index(drop=True, inplace=True)
                df_renewable.reset_index(drop=True, inplace=True)
                df_ev_demand.reset_index(drop=True, inplace=True)
                df_ev_soc.reset_index(drop=True, inplace=True)
                df_gen_p.reset_index(drop=True, inplace=True)
                df_gen_v.reset_index(drop=True, inplace=True)
                df_ev_action.reset_index(drop=True, inplace=True)
                df_voltage.reset_index(drop=True, inplace=True)
                df_line_loading.reset_index(drop=True, inplace=True)
                df_bus_violation.reset_index(drop=True, inplace=True)
                df_line_violation.reset_index(drop=True, inplace=True)
                df_phase_angle_violation.reset_index(drop=True, inplace=True)

            save_all_df_to_csv(random_seed, n_case, df_EV_spec, df_load_p, df_load_q, df_renewable, df_ev_demand, df_ev_soc, df_gen_p, df_gen_v, df_ev_action, df_voltage, df_line_loading, df_bus_violation, df_line_violation, df_phase_angle_violation) #, df_generation_cost)
            
            obs, info = eval_env.reset()
    eval_env.close()
    return rewards, RLTime, generation_cost, bus_violation, line_violation, angle_violation, SOCviolation

<<<<<<< HEAD
    return rewards, Time, Cost, N_violation, N_violation_relax

def save_all_df_to_csv(n_case, EVScenario, df_EV_spec, df_load_p, df_load_q, df_renewable, df_ev_demand, df_ev_soc, df_gen_p, df_gen_v, df_ev_action, df_voltage, df_line_loading, df_bus_violation, df_line_violation, df_phase_angle_violation): #, df_generation_cost):
=======
def save_all_df_to_csv(random_seed, n_case, df_EV_spec, df_load_p, df_load_q, df_renewable, df_ev_demand, df_ev_soc, df_gen_p, df_gen_v, df_ev_action, df_voltage, df_line_loading, df_bus_violation, df_line_violation, df_phase_angle_violation):
>>>>>>> origin/env/omit_EVScenario
    # Create the directory if it does not exist
        if not os.path.exists("Evaluation/Case%s/%s" %(n_case,random_seed)):
            os.makedirs("Evaluation/Case%s/%s" %(n_case,random_seed))

        # Save the dataframes to a csv file
        df_EV_spec.to_csv("Evaluation/Case%s/%s/EV_spec.csv" %(n_case,random_seed), index=False)
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
<<<<<<< HEAD
            "Evaluation/Case%s/Case%s_%s/phase_angle_violation.csv" %(n_case, n_case, EVScenario), index=False
        )

=======
            "Evaluation/Case%s/%s/phase_angle_violation.csv" %(n_case,random_seed), index=False
        )
>>>>>>> origin/env/omit_EVScenario


if __name__ == "__main__":

<<<<<<< HEAD
    EVScenarios = ["ImmediateFull", "ImmediateBalanced", "Home", "Night"]
    n_case = 9
    # n_case = int(input("Enter Test Case Number (9, 14, 30, 39, 57): "))
    sample_size = 2
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
            model = PPO.load("Training/Model/Case%s/Case%s_%s" %(n_case, n_case, EVScenario))
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
=======
    n_case = int(input("Enter Test Case Number (9, 14, 30, 57): "))

    sample_size = 30

    # check if the pkl exists
    if os.path.exists(f"Evaluation/Case{n_case}/RL_metrics_Case{n_case}.pkl"):
        with open(f"Evaluation/Case{n_case}/RL_metrics_Case{n_case}.pkl", "rb") as f:
            metrics = pickle.load(f)
            starting_sample = metrics["Sample"]
    else:
        starting_sample = 0
        metrics = {
            "Times": [],
            "Costs": [],
            "Total_rewards": [],
            "N_bus_violations": [],
            "N_line_violations": [],
            "N_angle_violations": [],
            "N_SOC_violations": []
        }

    for random_seed in range(starting_sample,sample_size+starting_sample):
        n_steps = 24
        model = PPO.load("Training/Model/Case%s/Case%s" %(n_case, n_case))
        model.set_random_seed(random_seed)
        rewards, Time, Cost, bus_violation, line_violation, angle_violation, SOCviolation = evaluate_PPO_model(
            n_steps=n_steps, n_case=n_case, model=model, random_seed=random_seed, generate_csv=True
        )
>>>>>>> origin/env/omit_EVScenario

        metrics["Times"].append(Time)
        metrics["Costs"].append(Cost)
        metrics["Total_rewards"].append(sum(rewards))
        metrics["N_bus_violations"].append(bus_violation)
        metrics["N_line_violations"].append(line_violation)
        metrics["N_angle_violations"].append(angle_violation)
        metrics["N_SOC_violations"].append(SOCviolation)
        
    
        print("Finished Sample: ", random_seed+1, "/", (starting_sample+sample_size))
       
    metrics["Sample"] = starting_sample + sample_size

    # Save the metrics dicts
    with open(f"Evaluation/Case{n_case}/RL_metrics_Case{n_case}.pkl" , "wb") as f:
        pickle.dump(metrics, f)