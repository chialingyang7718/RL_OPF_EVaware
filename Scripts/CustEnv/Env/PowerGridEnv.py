"""
This environment can be used to train a reinforcement learning agent to control the power grid. 
Assumptions:
- The environment is fully observable.
- The environment is deterministic.
- All the generators are static. (PQ bus)
- There is no min. requirement of PV or wind generation.
- There is no max. limitation of PV and wind curtailment.
- All the EVs have the same capacity and charging efficiency.
- All EV aggregators have the same number of max. allowed EVs. (100)
- All EV aggregators connect to the same busses as the loads.
- EV model: Tesla Model Y is chosen as the EV model since it is the best seller in the world in Jan. 2024. (https://cleantechnica.com/2024/03/05/top-selling-electric-vehicles-in-the-world-january-2024/)
- Actions: generation active power, generation reactive power (omitted ext. grid since it is typically not a direct control target)
- States: load active power, load reactive power, active power generation from PV or wind 
"""

#Import pandapower stuff
import pandapower as pp
import pandapower.toolbox as tb
import pandapower.control as control

#Import simbench stuff
import simbench as sb

#Import Gym stuff
from gymnasium import Env
from gymnasium.spaces import Box

#Import other stuff
import numpy as np
import random
import pandas as pd

#Create a custom environment
class PowerGrid(Env):
    def __init__(self, net, stdD = 0.03, dispatching_intervals = 96, UseDataSource = False, UseSimbench = False, EVaware = True):
        # inheritance from the parent class gymnasium.Env
        super(PowerGrid, self).__init__()

        # assign the grid
        self.net = net

        # convert the generator into static generator if there is any generator
        if net.gen.index.size > 0:
            self.net = self.change_gen_into_sgen(self.net)

        # read the number of gen & loads
        self.NL = self.net.load.index.size # NL: number of loads
        # self.NxG = self.net.ext_grid.index.size  # NxG: number of ext. grids
        self.NsG = self.net.sgen.index.size #NsG: number of static generators 
        

        # assign other parameters
        self.stdD = stdD # standard deviation of the consumption
        self.dispatching_intervals = dispatching_intervals #number of dispatching intervals
        self.UseDataSource = UseDataSource # whether to use the data source or random sampling
        self.UseSimbench = UseSimbench # whether to use the simbench data
        self.EVaware = EVaware # whether to consider the EV element

        # implement EVaware
        if self.EVaware == True:
            self.add_EV_storage(self.net)

        # initialization
        self.pre_reward = 0 #initialize the previous reward
        self.violation = False #initialize the violation
        self.episode_length = self.dispatching_intervals #initialize the episode length

        # define the action space: PsG, Qs, Pcharge, Pdischarge
        self.action_space = Box(low = np.full((2*self.NsG, ), -1), 
                                high = np.full((2*self.NsG, ), 1), 
                                shape=(2*self.NsG, ))
        ## Start from here
        # define the observation space: PL, QL, P_renewable(max generation from renewable energy sources)
        # ASSUMPTION: all the static generators are renewable energy sources
        self.observation_space = Box(low=np.zeros((2*self.NL+self.NsG, )), 
                                    high=np.full((2*self.NL+self.NsG, ), 1), 
                                    shape=(2*self.NL+self.NsG, ),
                                    dtype=np.float32)
        
        # assign the state based on whether to use the data source
        if self.UseDataSource == False:
            # assign generation limit, load limit, volatage limit, line limit manually
            self.define_limit_manually()
            # initialize the state with existing single data saved in the grid
            self.state = self.read_state_from_grid()
        else:
            # load the data source profiles
            self.load_profiles()
            # assign limits based on data source profiles
            self.define_limit_with_profile()
            # sampling the startpoint from the load profile and save the sampled profile into self.XX_profile
            self.load_ts_sampling()
            # update the state with the sampled profile
            self.state = np.concatenate((self.load_pmw_profile[0], 
                                        self.load_qvar_profile[0],
                                        self.re_pmw_profile[0]
                                        ), axis=None)


    def step(self, action):
        # intialize the terminated and truncated
        terminated = False
        truncated = False

        # apply state to load
        self.net = self.apply_state_to_load()

        # apply action to net
        self.net = self.apply_action_to_net(action)

        # run the power flow
        try:
            # pp.runpp(self.net, algorithm='gs')
            control.run_control(self.net, max_iter=30)
        except:
            # output the diagnostic report when the power flow does not converge
            print("Power flow does not converge!")
            print(pp.diagnostic(self.net, report_style='detailed')) 
            # assign the reward in the case of the power flow does not converge
            reward = -5000
        else:
            # calculate the reward in the case of the power flow converges
            reward = self.calculate_reward()
            # check if terminated in the case of no violation
            if self.violation == False:
                #terminate the episode if the reward is close to the previous reward
                terminated = abs(reward-self.pre_reward) <= 0.01
        # output the current episode length, reward, terminated, truncated
        print("episode length: ", self.episode_length, "Reward:", reward, "; Terminated:", terminated, "; Truncated:", truncated)
        # save the reward for the current step
        self.pre_reward = reward
        # decrease the episode length
        self.episode_length -= 1
        # check if the episode is terminated in the case of reaching the end of the episode
        terminated = self.episode_length == 0

        if terminated == False:
            # update the next state if the episode is not terminated
            if self.UseDataSource == False:
                 # add some noice
                for i in range(self.NL):
                    self.state[i] += self.stdD * (np.random.randn()) 
                    self.state[i+self.NL] += self.stdD * (np.random.randn()) 
                for i in range(self.NsG):
                    self.state[i + 2*self.NL] += self.stdD * (np.random.randn()) 
            else:
                # update with the sampled profile 
                self.state[0: self.NL] = self.load_pmw_profile[(self.dispatching_intervals - self.episode_length - 1)]
                self.state[self.NL: 2*self.NL] = self.load_qvar_profile[(self.dispatching_intervals - self.episode_length - 1)]
                self.state[2*self.NL:] = self.re_pmw_profile[(self.dispatching_intervals - self.episode_length - 1)]

        # get observation for the next state
        observation = self._get_observation()
        
        info = {}
        return observation, reward, terminated, truncated, info


    def render(self):
        pass


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # initialization
        self.pre_reward = 0 #initialize the previous reward
        self.violation = False #initialize the violation
        self.episode_length = self.dispatching_intervals #initialize the episode length

        # assign initial state of the upcoming episode
        if self.UseDataSource == False:
            # add some noice
            self.state += self.stdD * (np.random.randn(2*self.NL + self.NsG, ))
        else:
            # sampling the startpoint from the load profile
            self.load_ts_sampling()
            # update the state with the sampled profile
            self.state = np.concatenate((self.load_pmw_profile[0], 
                                         self.load_qvar_profile[0],
                                         self.re_pmw_profile[0]
                                         ), axis=None)

        # get observation of the states
        observation = self._get_observation() 
        info = {}
        return observation, info
    

    def close(self):
        print("Close this episode! The episode length is: ", self.episode_length)

    # convert the generation into static generator
    def change_gen_into_sgen(self, grid):
        for i in grid.gen.index:
            bus = grid.gen.loc[i, "bus"]
            p_mw = grid.gen.loc[i, "p_mw"]
            if "max_p_mw" not in grid.gen:
                max_p_mw = grid.gen.loc[i, "p_mw"] * 1.5 # assume the max power output is 1.5 times the current power output if not specified
            else:
                max_p_mw = grid.gen.loc[i, "max_p_mw"]
            if "min_p_mw" not in grid.gen:
                min_p_mw = grid.gen.loc[i, "p_mw"] * 0 # assume the min power output is 0 if not specified
            else:
                min_p_mw = grid.gen.loc[i, "min_p_mw"]
            if "max_q_mvar" not in grid.gen and "q_mvar" in grid.gen:
                max_q_mvar = grid.gen.loc[i, "q_mvar"] * 1.5 # assume the max reactive power output is 1.5 times the current reactive power output if not specified
            elif "q_mvar" not in grid.gen:
                max_q_mvar = 100 # assume the max reactive power output is 100 MVar
            else:
                max_q_mvar = grid.gen.loc[i, "max_q_mvar"]
            if "min_q_mvar" not in grid.gen and "q_mvar" in grid.gen:
                min_q_mvar = abs(grid.gen.loc[i, "q_mvar"]) * -1.5 # assume the min reactive power output is 1.5 times the current reactive power output if not specified
            elif "q_mvar" not in grid.gen:
                min_q_mvar = -100 # assume the min reactive power output is -100 MVar
            else:
                min_q_mvar = grid.gen.loc[i, "min_q_mvar"]
            grid.gen.drop(i, inplace = True)
            pp.create_sgen(grid, bus, p_mw=p_mw, q_mvar=0, max_p_mw=max_p_mw, min_p_mw=min_p_mw, max_q_mvar=max_q_mvar, min_q_mvar=min_q_mvar)
        return grid
    
    # add the EV element to the grid
    def add_EV_storage(self, grid):
        battery_capacity = 57.5/1000 # Tesla Model Y battery capacity in mWh
        n_car = 100 # number of EVs allowed in each EV aggregator # ASSUMPTION: 100 is the max. allowed number of EVs in each EV aggregator
        # add the EV storage to where the loads are
        for i in grid.load.index:
            bus = grid.load.loc[i, "bus"]
            pp.create_storage(grid, bus=bus, p_mw=0, max_e_mwh= battery_capacity*n_car, soc_percent=0, min_e_mwh=0)


    # [if UseDataSource == False] assign the generation limit, load limit, voltage limit, line limit manually    
    def define_limit_manually(self):
        # assign the static generator limits (PQ bus)
        if 'max_p_mw' not in self.net.sgen:
            self.net.sgen.loc[:,"max_p_mw"] = self.net.sgen.loc[:,"p_mw"] * 1.5 # assume the max power output is 1.5 times the current power output if not specified
        if 'max_q_mvar' not in self.net.sgen:
            self.net.sgen.loc[:,"max_q_mvar"] = abs(self.net.sgen.loc[:,"q_mvar"]) * 1.5 # assume the max reactive power output is 1.5 times the current reactive power output if not specified
        if 'min_q_mw' not in self.net.sgen:
            self.net.sgen.loc[:,"min_q_mvar"] = abs(self.net.sgen.loc[:,"q_mvar"]) * -1.5 # assume the min power output is 1.5 times the current power output if not specified
        # self.PsGmax = np.full(self.NsG, 10) # ASSUMPTION: the max power generation from static generator is 10 MW
        self.PsGmax = np.full(self.NsG, self.net.sgen.loc[:,'max_p_mw'].max()) # to avoid limitation of zero if any original value is zero
        self.PsGmin = np.zeros(self.NsG) # ASSUMPTION: the min power generation from static generator is 0 MW
        self.QsGmax = np.full(self.NsG,self.net.sgen.loc[:,'max_q_mvar'].max())
        self.QsGmin = np.full(self.NsG, self.net.sgen.loc[:,'min_q_mvar'].min())
        # assign the load limits (PQ bus)
        # self.PLmax = np.full(self.NL, 100) # ASSUMPTION: the max active power consumption is 100 MW
        self.PLmax = np.full(self.NL, self.net.load.loc[:, 'p_mw'].max() * 1.5) # ASSUMPTION: the max active power consumption is 1.5 times the maximum value in the net
        self.PLmin = np.zeros(self.NL) # ASSUMPTION: the min active power consumption is 0 MW
        # self.QLmax = np.full(self.NL, 100) # ASSUMPTION: the max reactive power consumption is 100 MVar
        self.QLmax = np.full(self.NL, abs(self.net.load.loc[:, 'q_mvar']).max() * 1.5) # ASSUMPTION: the max reactive power consumption is 1.5 times the maximum value in the net
        self.QLmin = -self.QLmax # ASSUMPTION: the reactive power consumption range is zero-centered
        # assign the voltage and line loading limits (valid for all buses)
        self.VGmin = 0.94 # ASSUMPTION: the min voltage is 0.94 pu
        self.VGmax = 1.06 # ASSUMPTION: the max voltage is 1.06 pu
        self.linemax = 100  # ASSUMPTION: the max line loading is 100%

    # [if UseDataSource == False] read the active and reactive power of the loads, active power of gen from self.net.load
    def read_state_from_grid(self):
        loads = self.net.load.loc[:, ['p_mw', 'q_mvar']].to_numpy() # reads the active and reactive power of the loads
        renewable = self.net.sgen.loc[:, ['p_mw']].to_numpy() # reads the active power of the static generators
        state = np.concatenate((loads.flatten("F").astype(np.float32), renewable.flatten("F").astype(np.float32)), axis=None)
        return state

    # [if UseDataSource == True] load the profiles from json file or simbench
    def load_profiles(self):
        if self.UseSimbench == False:
            # import the profile from json file
            self.profiles = pd.read_json("/Users/YANG_Chialing/Desktop/Master_Thesis_TUM/pandapower/tutorials/cigre_timeseries_15min.json") # profile for a day
        else:
            # import the profile from simbench
            self.profiles = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=True) # profiles for a year


    # [if UseDataSource == True & UseSimbench == True] assign the limits based on the profile
    def define_limit_with_profile(self):
        # assign the static generator limits (PQ bus)
        self.PsGmax = self.profiles[('sgen','p_mw')].max() * 1.5 # ASSUMPTION: the max power generation from static generator is 1.5 times the max value in the profile
        self.PsGmin = np.zeros(self.NsG) # ASSUMPTION: the min power generation from static generator is 0 MW
        self.QsGmax = np.full((self.NsG, ), 10) # ASSUMPTION: the max reactive power generation from static generator is 10 MVar
        self.QsGmin = -self.QsGmax # ASSUMPTION: the reactive power generation range is zero-centered
        # assign the load limits (PQ bus)
        self.PLmax = self.profiles[('load', 'p_mw')].max() * 10 # ASSUMPTION: the max active power consumption is 10 times the max value in the profile
        self.PLmin = np.zeros(self.NL) # ASSUMPTION: the min active power consumption is 0 MW
        self.QLmax = abs(self.profiles[('load', 'q_mvar')]).max() * 10 # ASSUMPTION: the max reactive power consumption is 10 times the max value in the profile
        self.QLmin = -self.QLmax # ASSUMPTION: the reactive power consumption range is zero-centered
        # assign the voltage and line loading limits (valid for all buses)
        self.VGmin = 0.94 # ASSUMPTION: the min voltage is 0.94 pu
        self.VGmax = 1.06 # ASSUMPTION: the max voltage is 1.06 pu
        self.linemax = 100 # ASSUMPTION: the max line loading is 100%

    # [if UseDataSource == True] randomly sample the starting time of the load profile
    def load_ts_sampling(self):
        # read the load active and reactive power prfiles, power generation profile
        load_pmw = self.profiles[('load', 'p_mw')]
        load_qvar = self.profiles[('load', 'q_mvar')]
        re_pmw = self.profiles[('sgen', 'p_mw')]
        # fetch the total time step of the profile 
        total_time_step = load_pmw.index.size
        # randomly sample the starting time
        sampled_starting_time = random.randint(0, total_time_step-1-self.dispatching_intervals)
        # save the sampled profile into self.XX_profile
        self.load_pmw_profile = load_pmw.loc[sampled_starting_time:
                                             sampled_starting_time+self.dispatching_intervals-1, :].to_numpy()
        self.load_qvar_profile = load_qvar.loc[sampled_starting_time:
                                               sampled_starting_time+self.dispatching_intervals-1, :].to_numpy()
        self.re_pmw_profile = re_pmw.loc[sampled_starting_time:
                                            sampled_starting_time+self.dispatching_intervals-1, :].to_numpy()
        
    # normalize the value to the range of space
    def normalize(self, value, max_value, min_value, max_space, min_space):
        denominator = (max_space - min_space) * (max_value - min_value)
        if denominator == 0:
            return 0
        else:
            return (value - min_value) / (max_value - min_value) * (max_space - min_space) + min_space

    # denormalize the space value to value
    def denormalize(self, value, max_value, min_value, max_space, min_space):
        denominator = (max_space - min_space) * (max_value - min_value)
        if denominator == 0:
            return 0
        else:
            return (value - min_space) / (max_space - min_space) * (max_value - min_value)  + min_value


    # convert state into obeservation with ASSUMPTION: the environment is fully observable
    def _get_observation(self): 
        normalized_obs = []
        for i in range(self.NL):
            normalized_obs.append(self.normalize(self.state[i], self.PLmax[i], self.PLmin[i], 1, 0))
        for i in range(self.NL):
            normalized_obs.append(self.normalize(self.state[i+self.NL], self.QLmax[i], self.QLmin[i], 1, 0))
        for i in range(self.NsG):
            normalized_obs.append(self.normalize(self.state[i+2*self.NL], self.PsGmax[i], self.PsGmin[i], 1, 0))
        return np.array(normalized_obs).astype(np.float32)

    # Apply the action to the net
    def apply_action_to_net(self, action):
        for i in range(self.NsG): # static generator (PQ bus)
            denormalized_p = self.denormalize(action[i], self.PsGmax[i], self.PsGmin[i], 1, -1)
            # if the action exceeds the max power generation from the renewable energy sources, then set the power generation to the max value
            if denormalized_p > self.state[i+2*self.NL]:
                self.net.sgen.loc[i,'p_mw'] = self.state[i+2*self.NL]
            else:
                self.net.sgen.loc[i,'p_mw'] = denormalized_p
            self.net.sgen.loc[i, 'q_mvar'] = self.denormalize(action[i+self.NsG], self.QsGmax[i], self.QsGmin[i], 1, -1)
        # for i in range(self.NxG): # external grid (slack bus)
        #     self.net.ext_grid.loc[i, 'vm_pu'] = self.denormalize(action[i+2*self.NsG], self.VGmax, self.VGmin, 1, -1)
        return self.net

    # Apply the state to the load
    def apply_state_to_load(self): 
        for i in self.net.load.index: # load (PQ bus)
            self.net.load.loc[i, 'p_mw'] = self.state[i]
            self.net.load.loc[i, 'q_mvar'] = self.state[i+self.NL]
        return self.net


    # calculate the reward 
    def calculate_reward(self):
        # check the violation
        violated_buses = tb.violated_buses(self.net, self.VGmin, self.VGmax) # bus voltage violation 
        violated_lines = tb.overloaded_lines(self.net, self.linemax) #line overload violation
        
        # assign rewards based on the violation condition and generation cost
        if violated_buses.size != 0 or violated_lines.size != 0 : 
            self.violation = True
            return -500*(len(violated_buses) + len(violated_lines))
        else:
            return 1000 - 0.1 * self.calculate_gen_cost() #TODO: reference to the paper

            
    # calculate the generation cost
    def calculate_gen_cost(self):
        gen_cost = 0
        #if the cost function is polynomial
        if self.net.poly_cost.index.size > 0:
            for i in self.net.sgen.index:
                gen_cost += self.net.res_sgen.p_mw[i]**2 * self.net.poly_cost.iat[i,4] + self.net.res_sgen.p_mw[i] * self.net.poly_cost.iat[i,3] + self.net.poly_cost.iat[i,2] \
                            + self.net.poly_cost.iat[i,5] + self.net.poly_cost.iat[i,6] * self.net.res_sgen.q_mvar[i] + self.net.poly_cost.iat[i,7] * self.net.res_sgen.q_mvar[i]**2
            for i in self.net.ext_grid.index:
                gen_cost += self.net.res_ext_grid.p_mw[i]**2 * self.net.poly_cost.iat[i,4] + self.net.res_ext_grid.p_mw[i] * self.net.poly_cost.iat[i,3] + self.net.poly_cost.iat[i,2] \
                            + self.net.poly_cost.iat[i,5] + self.net.poly_cost.iat[i,6] * self.net.res_ext_grid.q_mvar[i] + self.net.poly_cost.iat[i,7] * self.net.res_ext_grid.q_mvar[i]**2
        #if the cost function is piecewise linear        
        elif self.net.pwl_cost.index.size > 0:
            points_list = self.net.pwl_cost.at[i, 'points']
            for i in self.net.sgen.index:
                for points in points_list:
                    p0, p1, c01 = points
                    if p0 <= self.net.res_sgen.p_mw[i] < p1:
                        gen_cost += c01 * self.net.res_sgen.p_mw[i]
            for i in self.net.ext_grid.index:
                for points in points_list:
                    p0, p1, c01 = points
                    if p0 <= self.net.res_ext_grid.p_mw[i] < p1:
                        gen_cost += c01 * self.net.res_ext_grid.p_mw[i]
        #if the cost function is missing 
        else:
            total_gen = self.net.res_sgen['p_mw'].sum() + self.net.res_ext_grid['p_mw'].sum()
            gen_cost =  0.1 * total_gen**2 + 40 * total_gen
            # print("No cost function found for generators. Assume the cost function is 0.1 * p_tot**2 + 40 * p_tot.")
        return gen_cost
    