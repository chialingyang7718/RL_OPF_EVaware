"""
This environment can be used to train a reinforcement learning agent to control the power grid. 
Assumptions --- Grid and Generation
- All the generators PV bus.
- There is no min. requirement of power generation.
- There is no max. limitation of PV and wind curtailment.
- std. deviation of the random normal distribution is 1 for the load sampling.
_ The max. power capacity of the generators is uniformly distributed between PGmin and PGmax and updated every time step.

Assumptions --- EV
- The EVs are connected to the same bus as the loads.
- The number of EV in a EV group is the integer of nominal power of the loads.
- The EVs in the same group have the same spec (max_mwh, (dis)charging eff.) and driving profile
- Assume the EVs are connected in parallel.
- The maximum EV (dis)charging power is 22 kW.

Assumptions --- Environment
- One timestep represents one hour.
- The environment is deterministic.
- Actions: generation active power, generation voltage (omitted ext. grid since it is typically not a direct control target),
EV (dis)charging power
- States: load active power, load reactive power, active power generation from PV or wind, EV SOC, SOC threshold, EV connections
"""

# Import pandapower stuff
import pandapower as pp
import pandapower.toolbox as tb

# Import Gym stuff
from gymnasium import Env
from gymnasium.spaces import Box

# Import other stuff
import numpy as np
import random
import pandas as pd

# Import simbench stuff(unused)
import simbench as sb


# Create a custom environment
class PowerGrid(Env):
    def __init__(self, net, stdD=1, dispatching_intervals=24, EVScenario=None, Training=False):  # assuming dispatching intervals = number of hours in a day
        # inheritance from the parent class gymnasium.Env
        super(PowerGrid, self).__init__()

        # assign the grid
        self.net = net

        # convert any static generator into generator
        if net.sgen.index.size > 0:
            self.net = self.change_sgen_into_gen(self.net)

        # read the number of gen & loads
        self.NL = self.net.load.index.size  # NL: number of loads
        self.NG = self.net.gen.index.size  # NG: number of generators

        # assign other parameters
        self.stdD = stdD  # standard deviation of any random normal distribution
        self.dispatching_intervals = dispatching_intervals  # number of dispatching intervals
        self.EVScenario = EVScenario  # define the EV scenario
        self.training = Training
        self.episode_length = self.dispatching_intervals
        self.time_step = 0

        # implement EV element
        if self.EVScenario is not None:

            # load EV spec and driving profiles
            self.load_EV_spec_profiles()

            # initialize the EV paramenters
            self.N_EV = self.NL
            self.SOCviolation = 0
            self.EV_power_demand = np.zeros(self.N_EV)
            self.connect_EV_to_grid = np.ones(self.N_EV) # 1: connected, 0: disconnected
            
            # assign the EV group to the load buses
            if self.net.storage.index.size == 0:
                self.add_EV_group(self.net)
        else:
            self.N_EV = 0


        # define the action space: PG, VG, P_EV(positive for charging, negative for discharging)
        self.action_space = Box(
            low=np.full((2 * self.NG + self.N_EV,), -1),
            high=np.full((2 * self.NG + self.N_EV,), 1),
            shape=(2 * self.NG + self.N_EV,),
            dtype=np.float32,
        )

        # define the observation space: PL, QL, P_renewable(max generation from renewable energy sources), SOC_EV, SOC_threshold, connetion status
        self.observation_space = Box(
            low=np.zeros((2 * self.NL + self.NG + 3 * self.N_EV,)),
            high=np.full((2 * self.NL + self.NG + 3 * self.N_EV,), 1),
            shape=(2 * self.NL + self.NG + 3 * self.N_EV,),
            dtype=np.float32,
        )

        # assign state and action limits
        self.define_limit()

        # initialize the state with existing single data saved in the grid
        self.state = self.create_state()


    def step(self, action):
        # intialize the terminated and truncated
        terminated = False
        truncated = False

        # update net with action and state
        self.net = self.apply_state_to_load()
        self.net = self.apply_action_to_net(action)

        # run the power flow (Newton-Raphson method)
        try:
            pp.runpp(self.net)
            # pp.runpp(self.net, algorithm='gs')

        except:
            # output the diagnostic report when the power flow does not converge
            # print("Power flow does not converge!")
            # print(pp.diagnostic(self.net, report_style="detailed"))

            # assign the reward in the case of the power flow does not converge
            reward = -5000
            diverged = True

        else:
            diverged = False
            # calculate the reward in the case of the power flow converges
            reward, violated_buses, overload_lines, violated_phase = (
                self.calculate_reward()
            )


        # output the current episode length, reward, terminated, truncated
        # print("episode length: ", self.episode_length, "Reward:", reward, "; Terminated:", terminated, "; Truncated:", truncated)

        # save the reward for the current step
        # self.pre_reward = reward
        gen_cost = self.calculate_gen_cost()

        # update the info
        info = {
            "load_p": self.net.load.loc[:, ["p_mw"]],
            "load_q": self.net.load.loc[:, ["q_mvar"]],
            "renewable_p": self.PGcap,
            "generation_p": self.net.res_gen.loc[:, ["p_mw"]],
            "generation_q": self.net.res_gen.loc[:, ["q_mvar"]],
            "generation_v": self.net.res_gen.loc[:, ["vm_pu"]],
            "bus_voltage": self.net.res_bus.loc[:, ["vm_pu"]],
            "line_loading": self.net.res_line.loc[:, ["loading_percent"]],
            "generation_cost": gen_cost,
        }
        if not diverged:
            if violated_buses is not None:
                info["bus_violation"] = violated_buses
            else:
                info["bus_violation"] = []
            if overload_lines is not None:
                info["line_violation"] = overload_lines
            else:
                info["line_violation"] = []
            info["phase_angle_violation"] = violated_phase


        if self.EVScenario is not None:
            # record the EV related info
            info["EV_demand"] = self.EV_power_demand
            info["EV_p"] = self.net.res_storage.loc[:, "p_mw"]
            info["EV_SOC"] = self.net.storage.loc[:, "soc_percent"]
            info["soc_violation"] = self.SOCviolation
            info["SOC_threshold"] = self.df_EV.loc[(slice(None), self.time_step), "SOC"+self.EVScenario].values

        # decrease the episode length and update time step
        self.episode_length -= 1
        self.time_step = self.time_step + 1

        # check if the episode is terminated in the case of reaching the end of the episode
        terminated = self.episode_length == 0
        # if self.training == True:
        #     terminated = reward <= 0

        # update the next state if the episode is not terminated
        if terminated == False:

            # add some noice to load and renewable generation
            self.add_noice_load_renew_state()

            # update EV state for the next time step
            if self.EVScenario is not None:
                self.update_EV_SOC()
                self.update_EV_limit()
                self.update_consumption_connection_state()

        # get observation for the next state
        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_length = self.dispatching_intervals

        # assign initial states for load and generation states by adding noice
        self.add_noice_load_renew_state()

        # assign initial state for EV SOC
        if self.EVScenario is not None:
            # reselect another day for the EV profile randomly
            initial_soc, selected_start_time = self.select_randomly_day_EV_profile()
            # initialize the time step and soc of the EVs
            self.time_step = selected_start_time
            self.net.storage.loc[:, "soc_percent"] = initial_soc
            self.state[2 * self.NL + self.NG: 2 * self.NL + self.NG+ self.N_EV] = self.net.storage.loc[:, "soc_percent"]
            self.state[2 * self.NL + self.NG + self.N_EV: 2 * self.NL + self.NG+ 2 * self.N_EV] = self.df_EV.loc[(slice(None), self.time_step), "SOC"+self.EVScenario].to_numpy()

            # update the EV (dis)charging limit since SOC is changed
            self.update_EV_limit()
            self.update_consumption_connection_state()

            # update info with EV related info
            # update EV spec in the info
            info = {"EV_spec": self.net.storage,
                    "SOC_threshold":  self.net.storage.loc[:, "soc_percent"]}
                    

        # get observation of the states
        observation = self._get_observation()


        return observation, info

    def close(self):
        pass

    # convert the static generator into generator
    def change_sgen_into_gen(self, grid):
        for i in grid.sgen.index:
            bus = grid.sgen.loc[i, "bus"]
            p_mw = grid.sgen.loc[i, "p_mw"]
            if "max_p_mw" not in grid.sgen:
                max_p_mw = (
                    grid.sgen.loc[i, "p_mw"] * 1.5
                )  # assume the max power output is 1.5 times the current power output if not specified
            else:
                max_p_mw = grid.sgen.loc[i, "max_p_mw"]
            if "min_p_mw" not in grid.gen:
                min_p_mw = (
                    grid.sgen.loc[i, "p_mw"] * 0
                )  # assume the min power output is 0 if not specified
            else:
                min_p_mw = grid.sgen.loc[i, "min_p_mw"]
            if "max_q_mvar" not in grid.sgen and "q_mvar" in grid.sgen:
                max_q_mvar = (
                    grid.sgen.loc[i, "q_mvar"] * 1.5
                )  # assume the max reactive power output is 1.5 times the current reactive power output if not specified
            elif "q_mvar" not in grid.sgen:
                max_q_mvar = 100  # assume the max reactive power output is 100 MVar
            else:
                max_q_mvar = grid.sgen.loc[i, "max_q_mvar"]
            if "min_q_mvar" not in grid.sgen and "q_mvar" in grid.sgen:
                min_q_mvar = abs(grid.sgen.loc[i, "q_mvar"]) * -1.5
            elif "q_mvar" not in grid.sgen:
                min_q_mvar = -100  # assume the min reactive power output is -100 MVar
            else:
                min_q_mvar = grid.sgen.loc[i, "min_q_mvar"]
            grid.sgen.drop(i, inplace=True)
            pp.create_gen(
                grid,
                bus,
                p_mw=p_mw,
                vm_pu=1.0,
                max_p_mw=max_p_mw,
                min_p_mw=min_p_mw,
                max_q_mvar=max_q_mvar,
                min_q_mvar=min_q_mvar,
            )
        return grid

    # assign the state and action limits
    def define_limit(self):
        # assign the generator limits
        if "max_p_mw" not in self.net.gen:
            self.net.gen.loc[:, "max_p_mw"] = (
                self.net.gen.loc[:, "p_mw"] * 1.5
            )  # assume the max power output is 1.5 times the current power output if not specified
        if "min_p_mw" not in self.net.gen:
            self.net.gen.loc[:, "min_p_mw"] = (
                self.net.gen.loc[:, "p_mw"] * 0
            )  # assume the min power output is 0 if not specified
        if "max_q_mvar" not in self.net.gen:
            self.net.gen.loc[:, "max_q_mvar"] = (
                abs(self.net.gen.loc[:, "q_mvar"]) * 1.5
            )  # assume the max reactive power output is 1.5 times the current reactive power output if not specified
        if "min_q_mvar" not in self.net.gen:
            self.net.gen.loc[:, "min_q_mvar"] = (
                abs(self.net.gen.loc[:, "q_mvar"]) * -1.5
            )  # assume the min power output is 1.5 times the current power output if not specified
        self.PGmax = (
            self.net.gen.loc[:, "max_p_mw"].to_numpy().flatten("F").astype(np.float64)
        )
        self.PGmin = (
            self.net.gen.loc[:, "min_p_mw"].to_numpy().flatten("F").astype(np.float64)
        )
        self.QGmax = (
            self.net.gen.loc[:, "max_q_mvar"].to_numpy().flatten("F").astype(np.float64)
        )
        self.QGmin = (
            self.net.gen.loc[:, "min_q_mvar"].to_numpy().flatten("F").astype(np.float64)
        )
        # assign the voltage limits
        self.Vmax = (
            self.net.bus.loc[:, "max_vm_pu"].to_numpy().flatten("F").astype(np.float64)
        )
        self.Vmin = (
            self.net.bus.loc[:, "min_vm_pu"].to_numpy().flatten("F").astype(np.float64)
        )
        self.VGmax = np.array([], dtype=np.float64)
        self.VGmin = np.array([], dtype=np.float64)
        for i in self.net.gen.loc[:, "bus"]:
            self.VGmax = np.append(self.VGmax, self.Vmax[i])
            self.VGmin = np.append(self.VGmin, self.Vmin[i])

        # assign the line limits
        self.linemax = (
            self.net.line.loc[:, "max_loading_percent"].to_numpy().flatten("F").astype(np.float64)
        )
        # assign phase angle difference limits
        self.theta_max = 30  # ASSUMPTION: the max phase angle difference is 30 degrees


        # assign the load limits (PQ bus)
        PLmax = (self.net.load.loc[:, "p_mw"] * 1.4)  # ASSUMPTION: the max active power consumption is 1.4 times the default value in the net
        PLmin = self.net.load.loc[:, "p_mw"] * 0.6  # ASSUMPTION: the min active power consumption is 0.6 times the default value in the net
        self.PLmax = PLmax.to_numpy().flatten("F").astype(np.float64)
        self.PLmin = PLmin.to_numpy().flatten("F").astype(np.float64)
        QLmax = (abs(self.net.load.loc[:, "q_mvar"]) * 1.4)  
        self.QLmax = QLmax.to_numpy().flatten("F").astype(np.float64)
        self.QLmin = (-self.QLmax)  # ASSUMPTION: the reactive power consumption range is zero-centered


        # assign the EV limits
        if self.EVScenario is not None:
            # initialize the EV charging limit
            self.PEVmax = np.zeros(self.N_EV)
            self.PEVmin = np.zeros(self.N_EV)

            # update the EV (dis)charging limit
            self.update_EV_limit()

    # update EV (dis)charging limit
    def update_EV_limit(self):
        discharge_max = (self.net.storage.loc[:, "max_e_mwh"] * self.net.storage.loc[:, "soc_percent"])
        charging_max = self.net.storage.loc[:, "max_e_mwh"] * (1 - self.net.storage.loc[:, "soc_percent"])
        for i in range(self.N_EV):
            # fetch EV power demand from the EV profile
            EV_power_demand = (
                self.df_EV.loc[(i, self.time_step), "DrivingConsumption_kWh"]
                * self.net.storage.loc[i, "n_car"]
                / 1000
            )  # power requirement from cars in MW
            # reserve the power for EV demand (to avoid discharging too much that EVs demand is not fullfilled)
            discharge_max[i] -= EV_power_demand
            # the charging speed is 22 kw and assume all the EVs are connected in parallel
            self.PEVmax[i] = min(0.001 * 22 * self.net.storage.loc[i, "n_car"], charging_max[i])
            self.PEVmin[i] = -min(0.001 * 22 * self.net.storage.loc[i, "n_car"], discharge_max[i])
    

    def update_consumption_connection_state(self):
        for i in range(self.N_EV):
            # fetch EV power demand from the EV profile
            EV_power_demand = (
                self.df_EV.loc[(i, self.time_step), "DrivingConsumption_kWh"]
                * self.net.storage.loc[i, "n_car"]
                / 1000
            )  # power requirement from cars in MW
            self.EV_power_demand[i] = EV_power_demand

            # update the connection state
            if self.EV_power_demand[i] != 0:
                self.connect_EV_to_grid[i] = 0
                self.net.storage.loc[i, "in_service"] = False
            else:
                self.connect_EV_to_grid[i] = 1
                self.net.storage.loc[i, "in_service"] = True
        self.state[self.NL * 2 + self.NG + 2 * self.N_EV: ] = self.connect_EV_to_grid

    def add_noice_load_renew_state(self):
        # add some noice to PL and QL
        # for i in self.net.load.index:
        #     while True:
        #         random_PL = np.random.normal(self.mu_PL[i], self.stdD)
        #         if self.PLmin[i] <= random_PL <= self.PLmax[i]:
        #             self.state[i] = random_PL
        #             break
        #     while True:
        #         random_QL = np.random.normal(self.mu_QL[i], self.stdD)
        #         if self.QLmin[i] <= random_QL <= self.QLmax[i]:
        #             self.state[i + self.NL] = random_QL
        #             break
        self.state[: self.NL] = np.random.uniform(self.PLmin, self.PLmax)
        self.state[self.NL : 2 * self.NL] = np.random.uniform(self.QLmin, self.QLmax)

        # add some noice to PGcap
        self.PGcap = np.random.uniform(self.PGmin, self.PGmax)
        self.state[2 * self.NL : 2 * self.NL + self.NG] = self.PGcap


    # read the active and reactive power of the loads, active power of gen from self.net.load
    def create_state(self):
        # read the active and reactive power of the loads
        self.mu_PL = self.net.load.loc[:, ["p_mw"]].to_numpy()
        self.mu_QL = self.net.load.loc[:, ["q_mvar"]].to_numpy()
        self.PGcap = np.random.uniform(self.PGmin, self.PGmax)
        # concatenate state
        if self.EVScenario is None:
            state = np.concatenate(
                (
                    self.mu_PL.flatten("F").astype(np.float32),
                    self.mu_QL.flatten("F").astype(np.float32),
                    self.PGcap,
                ),
                axis=None,
            )
        else:
            # randomly select SOC of the EVs under defined EV scenario as the initial SOC
            initial_soc, selected_start_time = self.select_randomly_day_EV_profile()
            self.time_step = selected_start_time
            self.net.storage.loc[:, "soc_percent"] = initial_soc

            for i in range(self.N_EV):
                # update the connection state
                if self.EV_power_demand[i] != 0:
                    self.connect_EV_to_grid[i] = 0
                    self.net.storage.loc[i, "in_service"] = False
                else:
                    self.connect_EV_to_grid[i] = 1
                    self.net.storage.loc[i, "in_service"] = True

            state = np.concatenate(
                (
                    self.mu_PL.flatten("F").astype(np.float32),
                    self.mu_QL.flatten("F").astype(np.float32),
                    self.PGcap,
                    initial_soc.flatten("F").astype(np.float32),
                    initial_soc.flatten("F").astype(np.float32),
                    self.connect_EV_to_grid
                ),
                axis=None,
            )
        return state

    def update_EV_SOC(self):
        for i in range(self.N_EV):

            # calculate the energy before (dis)charging
            energy_b4 = (
                self.net.storage.loc[i, "max_e_mwh"]
                * self.net.storage.loc[i, "soc_percent"]
            )

            # # fetch charging efficiency from the EV spec
            # df_charging_eff = self.df_EV_spec[
            #     self.df_EV_spec["Parameter"] == "Battery_charging_efficiency"
            # ].set_index("Parameter")
            # charging_efficiency = df_charging_eff[str(i)].values
            # self.net.storage.loc[i, "eta_c"] = charging_efficiency
        

            # update the SOC of the EVs
            # if (self.net.res_storage.loc[i, "p_mw"] >= 0):  # charging with time step 1 hour
            #     energy_aftercharging = (
            #         energy_b4
            #         + self.net.res_storage.loc[i, "p_mw"] * charging_efficiency * 1
            #         - EV_power_demand * 1
            #     )
            energy_after = energy_b4 + self.net.res_storage.loc[i, "p_mw"] - self.EV_power_demand[i] 
        
            if energy_after > self.net.storage.loc[i, "max_e_mwh"] :
                print("Overcharging!")
                self.net.storage.loc[i, "soc_percent"] = 1
            elif energy_after < 0:
                print("Negative SOC!")
                self.net.storage.loc[i, "soc_percent"] = 0
            else:
                self.net.storage.loc[i, "soc_percent"] = (
                    energy_after / self.net.storage.loc[i, "max_e_mwh"]
                )
            # else:  # discharging with time step 1 hour
            #     energy_afterdischarging = (
            #         energy_b4
            #         + self.net.res_storage.loc[i, "p_mw"] * 1
            #         - EV_power_demand * 1
            #     )
            #     if energy_afterdischarging >= 0:
            #         self.net.storage.loc[i, "soc_percent"] = (
            #             energy_afterdischarging / self.net.storage.loc[i, "max_e_mwh"]
            #         )
            #     else:
            #         self.net.storage.loc[i, "soc_percent"] = 0
            #         print("Negative SOC!")
            self.state[self.NL * 2 + self.NG + i] = self.net.storage.loc[i, "soc_percent"]
            
        self.state[self.NL * 2 + self.NG + self.N_EV: self.NL * 2 + self.NG + 2 * self.N_EV] = self.df_EV.loc[(slice(None), self.time_step), "SOC"+self.EVScenario].to_numpy()

        

    def load_EV_spec_profiles(self):
        # Load the EV specification
        rows_to_read = [
            0,
            1069,
            1070,
            1071,
        ]  # Adjust for 0-based indexing, considering the header as row 0
        self.df_EV_spec = pd.read_csv(
            "Data/German_EV/emobpy_input_data.csv",
            skiprows=lambda x: x not in rows_to_read and x != 0,
            header=0,
        )
        self.df_EV_spec.drop(columns=self.df_EV_spec.columns[0:4], inplace=True)
        self.df_EV_spec.rename(columns={"Level_3": "Parameter"}, inplace=True)

        # load the EV driving profile
        self.df_EV = pd.read_csv(
            "Data/German_EV/emobpy_timeseries_hourly.csv", low_memory=False
        )

        # Following is the preprossing of df_EV:
        # Modify column names by appending the content of the first row
        new_column_names = [
            f"{col}_{self.df_EV.at[0, col]}" for col in self.df_EV.columns
        ]

        # Assign the new column names to the DataFrame
        self.df_EV.columns = new_column_names

        # Drop the first and second rows
        self.df_EV = self.df_EV.drop([0, 1]).reset_index(drop=True)

        # Define a conversion function to convert values into float or int
        def convert_to_float_int(value):
            try:
                # Try to convert to float
                float_val = float(value)
                # If the float value is equivalent to an int, return as int
                if float_val.is_integer():
                    return int(float_val)
                else:
                    return float_val
            except ValueError:
                # Return the original value if conversion fails
                return value

        # Apply the conversion function to each element in the DataFrame
        self.df_EV = self.df_EV.map(convert_to_float_int)
        self.df_EV = self.df_EV.rename(
            columns={
                "Unnamed: 0_nan": "Time",
                "ID_ID": "ID",
                "VehicleMobility_Distance_km": "Distance_km",
                "DrivingConsumption_Consumption_kWh": "DrivingConsumption_kWh",
                "GridAvailability_PowerRating_kW": "ChargingAvailability_kW",
                "GridDemand_Immediate_full_capacity_Load_kW": "ChargingPowerImmediateFull_kW",
                "GridDemand_Immediate_full_capacity.1_SoC": "SOCImmediateFull",
                "GridDemand_Immediate_balanced_Load_kW": "ChargingPowerImmediateBalanced_kW",
                "GridDemand_Immediate_balanced.1_SoC": "SOCImmediateBalanced",
                "GridDemand_From_0_to_24_at_home_Load_kW": "ChargingPowerHome_kW",
                "GridDemand_From_0_to_24_at_home.1_SoC": "SOCHome",
                "GridDemand_From_23_to_8_at_home_Load_kW": "ChargingPowerNight_kW",
                "GridDemand_From_23_to_8_at_home.1_SoC": "SOCNight",
            }
        )
        self.df_EV = self.df_EV[self.df_EV["ID"] < self.net.load.index.size]
        self.df_EV["Time"] = pd.to_datetime(self.df_EV["Time"])
        self.df_EV["Time_step"] = (
            self.df_EV["Time"].astype("int64")
            - self.df_EV["Time"].astype("int64").min()
        ) // (3600 * 10**9)
        self.df_EV.set_index(["ID", "Time_step"], inplace=True)
        # Now the df_EV is a two-level indexed DataFrame with the ID and Time_step as the indices in which only relevant IDs are extracted.
        # Notice that all days are included. Thus, the total time steps are 365*24 = 8760.

    def add_EV_group(self, grid):
        # add the EV group to where the loads are
        for i in grid.load.index:
            bus = grid.load.loc[i, "bus"]
            n_car = int(grid.load.loc[i, "p_mw"])  # number of EVs connected to the bus is assumed to be integer of nominal power of the loads
            if n_car == 0:
                n_car = 1 # if the nominal power of the load is close to zero, assume one EV is connected
            pp.create_storage(
                grid,
                bus=bus,
                p_mw=0,
                max_e_mwh = self.df_EV_spec.loc[0, str(i)] * n_car / 1000,
                soc_percent=0.5,
                min_e_mwh=0,
                evid=i,
                n_car=n_car,
            )

            

    def select_randomly_day_EV_profile(self):
        # randomly select the EV profile from the df_EV
        selected_time = random.randint(0, 364) * 24
        df_SOC_threshold = self.df_EV.loc[(slice(None), slice(selected_time, selected_time + 23)), "SOC"+self.EVScenario]
        inital_soc = df_SOC_threshold.groupby(level='ID').first().to_numpy()
        return inital_soc, selected_time

    # normalize the value to the range of space
    def normalize(self, value, max_value, min_value, max_space, min_space):
        denominator = (max_space - min_space) * (max_value - min_value)
        if denominator == 0:
            return 0
        else:
            return (value - min_value) / (max_value - min_value) * (
                max_space - min_space
            ) + min_space

    # denormalize the space value to value
    def denormalize(self, value, max_value, min_value, max_space, min_space):
        denominator = (max_space - min_space) * (max_value - min_value)
        if denominator == 0:
            return 0
        else:
            return (value - min_space) / (max_space - min_space) * (
                max_value - min_value
            ) + min_value

    # convert state into obeservation with normalization
    def _get_observation(self):
        normalized_obs = []
        for i in range(self.NL):
            normalized_obs.append(
                self.normalize(self.state[i], self.PLmax[i], self.PLmin[i], 1, 0)
            )

        for i in range(self.NL):
            normalized_obs.append(
                self.normalize(
                    self.state[i + self.NL], self.QLmax[i], self.QLmin[i], 1, 0
                )
            )

        for i in range(self.NG):
            normalized_obs.append(
                self.normalize(
                    self.state[i + 2 * self.NL], self.PGmax[i], self.PGmin[i], 1, 0
                )
            )

        # directly assign the SOC of the EVs to the observation
        if self.EVScenario is not None:
            normalized_obs.extend(self.state[self.NL * 2 + self.NG :])

        return np.array(normalized_obs).astype(np.float32)

    # apply the action to the net
    def apply_action_to_net(self, action):
        for i in range(self.NG):  # Generator (PV bus)
            denormalized_p = self.denormalize(
                action[i], self.PGcap[i], self.PGmin[i], 1, -1
            )
            # # set the power generation to the max value if the action is out of the limit
            if denormalized_p > self.state[i + 2 * self.NL]:
                self.net.gen.loc[i, "p_mw"] = self.state[i + 2 * self.NL]
            else:
                self.net.gen.loc[i, "p_mw"] = denormalized_p
            self.net.gen.loc[i, "vm_pu"] = self.denormalize(
                action[i + self.NG], self.VGmax[i], self.VGmin[i], 1, -1
            )

            for i in range(self.N_EV):
                denormalized_pEV = self.denormalize(
                    action[i + 2 * self.NG], self.PEVmax[i], self.PEVmin[i], 1, -1
                )
                # discharge_efficiency = self.df_EV_spec[
                #     self.df_EV_spec["Parameter"] == "Battery_discharging_efficiency"
                # ][str(i)].values
                # self.net.storage.loc[i, "eta_d"] = discharge_efficiency
                self.net.storage.loc[i, "p_mw"] = denormalized_pEV
                # # set the EV power based on the action
                # if denormalized_pEV < 0:
                #     # discharging
                #     self.net.storage.loc[i, "p_mw"] = (discharge_efficiency * denormalized_pEV)
                # else:
                #     # charging
                #     self.net.storage.loc[i, "p_mw"] = denormalized_pEV
        return self.net

    # Apply the state to the load
    def apply_state_to_load(self):
        for i in self.net.load.index:  # load (PQ bus)
            self.net.load.loc[i, "p_mw"] = self.state[i]
            self.net.load.loc[i, "q_mvar"] = self.state[i + self.NL]
        return self.net

    # calculate the reward
    def calculate_reward(self):
        # check the violation in the grid and assign penalty
        violated_buses = tb.violated_buses(self.net, self.Vmin, self.Vmax)
        overload_lines = tb.overloaded_lines(self.net, self.linemax)
        violated_phase = []
        penalty_voltage = 0
        penalty_line = 0
        penalty_phase_angle = 0
        self.violation = False

        # bus voltage violation
        if violated_buses.size != 0:
            self.violation = True
            for violated_bus in violated_buses:
                if self.net.res_bus.loc[violated_bus, "vm_pu"] < self.Vmin[violated_bus]:
                    penalty_voltage += (
                        self.net.res_bus.loc[violated_bus, "vm_pu"] - self.Vmin[violated_bus]
                    ) * 1000
                else:
                    penalty_voltage += (
                        self.Vmax[violated_bus] - self.net.res_bus.loc[violated_bus, "vm_pu"]
                    ) * 1000

        # line overload violation
        if overload_lines.size != 0:
            self.violation = True
            for overload_line in overload_lines:
                penalty_line += (
                    self.linemax[overload_line] - self.net.res_line.loc[overload_line, "loading_percent"]
                ) * 10

        # phase angle violation
        self.net.res_line["angle_diff"] = abs(self.net.res_line["va_from_degree"] - self.net.res_line["va_to_degree"])
        for i in self.net.res_line.index:
            if self.net.res_line["angle_diff"][i] > self.theta_max:
                self.violation = True
                violated_phase.append(i)
                penalty_phase_angle += (self.theta_max - self.net.res_line.loc[i, "angle_diff"]) * 10

        # EV SOC violation
        if self.EVScenario is not None:
            penalty_EV = 0
            for i in range(self.N_EV):
                energy_b4 = (self.net.storage.loc[i, "max_e_mwh"]* self.net.storage.loc[i, "soc_percent"])
                energy_after = energy_b4 + self.net.res_storage.loc[i, "p_mw"] - self.EV_power_demand[i] 
                updated_SOC = energy_after / self.net.storage.loc[i, "max_e_mwh"]
                SOC_threshold = self.df_EV.loc[(i, self.time_step), "SOC"+self.EVScenario]
                if SOC_threshold > updated_SOC:
                    self.violation = True
                    self.SOCviolation = 1
                    penalty_EV += (updated_SOC - SOC_threshold) * 1000
                elif updated_SOC > 1:
                    self.violation = True
                    self.SOCviolation = 1
                    penalty_EV += (1 - updated_SOC) * 1000
                else:
                    self.SOCviolation = 0

        # assign rewards based on the violation condition and generation cost
        if self.violation == True:
            reward = penalty_voltage + penalty_line + penalty_EV + penalty_phase_angle
        else:
            reward = 1000 - 0.1 * self.calculate_gen_cost()

        return reward, violated_buses, overload_lines, violated_phase

    # calculate the generation cost
    def calculate_gen_cost(self):
        gen_cost = 0
        # if the cost function is polynomial
        if self.net.poly_cost.index.size > 0:
            costfunction_generator = self.net.poly_cost[
                self.net.poly_cost["et"] == "gen"
            ]
            costfunction_extgrid = self.net.poly_cost[
                self.net.poly_cost["et"] == "ext_grid"
            ]
            for i in self.net.gen.index:
                gen_cost += (
                    self.net.res_gen.p_mw[i] ** 2 * costfunction_generator.iat[i, 4]
                    + self.net.res_gen.p_mw[i] * costfunction_generator.iat[i, 3]
                    + costfunction_generator.iat[i, 2]
                    + costfunction_generator.iat[i, 5]
                    + costfunction_generator.iat[i, 6] * self.net.res_gen.q_mvar[i]
                    + costfunction_generator.iat[i, 7] * self.net.res_gen.q_mvar[i] ** 2
                )
            for i in self.net.ext_grid.index:
                gen_cost += (
                    self.net.res_ext_grid.p_mw[i] ** 2 * costfunction_extgrid.iat[i, 4]
                    + self.net.res_ext_grid.p_mw[i] * costfunction_extgrid.iat[i, 3]
                    + costfunction_extgrid.iat[i, 2]
                    + costfunction_extgrid.iat[i, 5]
                    + costfunction_extgrid.iat[i, 6] * self.net.res_ext_grid.q_mvar[i]
                    + costfunction_extgrid.iat[i, 7]
                    * self.net.res_ext_grid.q_mvar[i] ** 2
                )
        # if the cost function is piecewise linear
        elif self.net.pwl_cost.index.size > 0:
            points_list = self.net.pwl_cost.at[i, "points"]
            for i in self.net.gen.index:
                for points in points_list:
                    p0, p1, c01 = points
                    if p0 <= self.net.res_gen.p_mw[i] < p1:
                        gen_cost += c01 * self.net.res_gen.p_mw[i]
            for i in self.net.ext_grid.index:
                for points in points_list:
                    p0, p1, c01 = points
                    if p0 <= self.net.res_ext_grid.p_mw[i] < p1:
                        gen_cost += c01 * self.net.res_ext_grid.p_mw[i]
        # if the cost function is missing
        else:
            total_gen = (
                self.net.res_gen["p_mw"].sum() + self.net.res_ext_grid["p_mw"].sum()
            )
            gen_cost = 0.1 * total_gen**2 + 40 * total_gen
            # print("No cost function found for generators. Assume the cost function is 0.1 * p_tot**2 + 40 * p_tot.")
        return gen_cost

    # fetch SOC state
    def get_soc(self):
        return self.net.storage["soc_percent"].values

    ####################################
    # Unused Code
    ####################################

    # convert the generator into static generator
    def change_gen_into_sgen(self, grid):
        for i in grid.gen.index:
            bus = grid.gen.loc[i, "bus"]
            p_mw = grid.gen.loc[i, "p_mw"]
            if "max_p_mw" not in grid.gen:
                max_p_mw = (
                    grid.gen.loc[i, "p_mw"] * 1.5
                )  # assume the max power output is 1.5 times the current power output if not specified
            else:
                max_p_mw = grid.gen.loc[i, "max_p_mw"]
            if "min_p_mw" not in grid.gen:
                min_p_mw = (
                    grid.gen.loc[i, "p_mw"] * 0
                )  # assume the min power output is 0 if not specified
            else:
                min_p_mw = grid.gen.loc[i, "min_p_mw"]
            if "max_q_mvar" not in grid.gen and "q_mvar" in grid.gen:
                max_q_mvar = (
                    grid.gen.loc[i, "q_mvar"] * 1.5
                )  # assume the max reactive power output is 1.5 times the current reactive power output if not specified
            elif "q_mvar" not in grid.gen:
                max_q_mvar = 100  # assume the max reactive power output is 100 MVar
            else:
                max_q_mvar = grid.gen.loc[i, "max_q_mvar"]
            if "min_q_mvar" not in grid.gen and "q_mvar" in grid.gen:
                min_q_mvar = (
                    abs(grid.gen.loc[i, "q_mvar"]) * -1.5
                )  # assume the min reactive power output is 1.5 times the current reactive power output if not specified
            elif "q_mvar" not in grid.gen:
                min_q_mvar = -100  # assume the min reactive power output is -100 MVar
            else:
                min_q_mvar = grid.gen.loc[i, "min_q_mvar"]
            grid.gen.drop(i, inplace=True)
            pp.create_sgen(
                grid,
                bus,
                p_mw=p_mw,
                q_mvar=0,
                max_p_mw=max_p_mw,
                min_p_mw=min_p_mw,
                max_q_mvar=max_q_mvar,
                min_q_mvar=min_q_mvar,
            )
        return grid

    # assign the state based on the data source
    def assign_state_with_datasource(self):
        # load the data source profiles
        self.load_profiles()
        # assign limits based on data source profiles
        self.define_limit_with_profile()
        # sampling the startpoint from the load profile and save the sampled profile into self.XX_profile
        self.load_ts_sampling()
        # update the state with the sampled profile
        self.state = np.concatenate(
            (
                self.load_pmw_profile[0],
                self.load_qvar_profile[0],
                self.re_pmw_profile[0],
            ),
            axis=None,
        )

    # update the state with the sampled profile
    def update_state_with_datasource(self):
        self.state[0 : self.NL] = self.load_pmw_profile[
            (self.dispatching_intervals - self.episode_length - 1)
        ]
        self.state[self.NL : 2 * self.NL] = self.load_qvar_profile[
            (self.dispatching_intervals - self.episode_length - 1)
        ]
        self.state[2 * self.NL :] = self.re_pmw_profile[
            (self.dispatching_intervals - self.episode_length - 1)
        ]

    def load_profiles(self, UseSimbench):
        if UseSimbench == False:
            # import the profile from json file
            self.profiles = pd.read_json(
                "/Users/YANG_Chialing/Desktop/Master_Thesis_TUM/pandapower/tutorials/cigre_timeseries_15min.json"
            )  # profile for a day
        else:
            # import the profile from simbench
            self.profiles = sb.get_absolute_values(
                self.net, profiles_instead_of_study_cases=True
            )  # profiles for a year

    # assign the limits based on the datasource profile
    def define_limit_with_profile(self):
        # assign the static generator limits (PQ bus)
        self.PGmax = (
            self.profiles[("sgen", "p_mw")].max() * 1.5
        )  # ASSUMPTION: the max power generation from static generator is 1.5 times the max value in the profile
        self.PGmin = np.zeros(
            self.NG
        )  # ASSUMPTION: the min power generation from static generator is 0 MW
        self.QGmax = np.full(
            (self.NG,), 10
        )  # ASSUMPTION: the max reactive power generation from static generator is 10 MVar
        self.QGmin = (
            -self.QGmax
        )  # ASSUMPTION: the reactive power generation range is zero-centered
        # assign the load limits (PQ bus)
        self.PLmax = (
            self.profiles[("load", "p_mw")].max() * 10
        )  # ASSUMPTION: the max active power consumption is 10 times the max value in the profile
        self.PLmin = np.zeros(
            self.NL
        )  # ASSUMPTION: the min active power consumption is 0 MW
        self.QLmax = (
            abs(self.profiles[("load", "q_mvar")]).max() * 10
        )  # ASSUMPTION: the max reactive power consumption is 10 times the max value in the profile
        self.QLmin = (
            -self.QLmax
        )  # ASSUMPTION: the reactive power consumption range is zero-centered
        # assign the voltage and line loading limits (valid for all buses)
        self.VGmin = 0.94  # ASSUMPTION: the min voltage is 0.94 pu
        self.VGmax = 1.06  # ASSUMPTION: the max voltage is 1.06 pu
        self.linemax = 100  # ASSUMPTION: the max line loading is 100%

    # randomly sample the starting time of the load profile
    def load_ts_sampling(self):
        # read the load active and reactive power prfiles, power generation profile
        load_pmw = self.profiles[("load", "p_mw")]
        load_qvar = self.profiles[("load", "q_mvar")]
        re_pmw = self.profiles[("sgen", "p_mw")]
        # fetch the total time step of the profile
        total_time_step = load_pmw.index.size
        # randomly sample the starting time
        sampled_starting_time = random.randint(
            0, total_time_step - 1 - self.dispatching_intervals
        )
        # save the sampled profile into self.XX_profile
        self.load_pmw_profile = load_pmw.loc[
            sampled_starting_time : sampled_starting_time
            + self.dispatching_intervals
            - 1,
            :,
        ].to_numpy()
        self.load_qvar_profile = load_qvar.loc[
            sampled_starting_time : sampled_starting_time
            + self.dispatching_intervals
            - 1,
            :,
        ].to_numpy()
        self.re_pmw_profile = re_pmw.loc[
            sampled_starting_time : sampled_starting_time
            + self.dispatching_intervals
            - 1,
            :,
        ].to_numpy()

    # assign EV to load buses if we decide to differentiate the EV driving profile within the same group
    def assign_EV_to_loads(self, grid):
        total_EV = (
            self.NL * 10
        )  # total number of EVs is assumed to be 10 times the number of loads
        EV_ID = []

        # randomly extract EV ID list range from 0 to 199
        for i in range(total_EV):
            EV_ID.append(random.randint(0, 199))

        # assign the EV(ID) to the loads depending on the nominal power of the loads. If there are EV(ID) left, assign them to the loads with the highest nominal power
        load_sum = grid.load.loc[:, "p_mw"].sum()
        grid.load["EV_ID"] = [
            [] for _ in grid.load.index
        ]  # Initialize each cell with an empty list
        for i in grid.load.index:
            weight = grid.load.loc[i, "p_mw"] / load_sum
            n_EV_bus = int(weight * total_EV)  # number of EVs connected to the bus

            for _ in range(n_EV_bus):
                if EV_ID:  # Check if there are still EV_IDs available
                    grid.load.loc[i, "EV_ID"].append(
                        EV_ID.pop(0)
                    )  # Assign and remove the first available EV_ID
                else:
                    break  # No more EV_IDs to assign

            # assign the remaining EV_IDs to the load with the highest nominal power
            max_load = grid.load["p_mw"].idxmax()
            if EV_ID:
                grid.load.loc[max_load, "EV_ID"].extend(EV_ID)
