#Import pandapower stuff
import pandapower as pp
import pandapower.networks as pn
import pandapower.toolbox as tb
import pandapower.control as control


#Import Gym stuff
from gymnasium import Env
from gymnasium.spaces import Box


#Import stable baselines stuff
from stable_baselines3 import PPO


#Import other stuff
import numpy as np
import random
import os
import pandas as pd



#Create a custom environment
class PowerGridEnv(Env):
    def __init__(self, net, stdD, dispatching_intervals, UseDataSource):
        super(PowerGridEnv, self).__init__()
        # assign the grid
        self.net = net

        # read the number of gen & loads
        self.NL = self.net.load.index.size # NL: number of loads
        self.NG = self.net.gen.index.size # NG: number of generators 
        self.NxG = self.net.ext_grid.index.size  # NxG: number of ext. grids
        self.NsG = self.net.sgen.index.size #NsG: number of static generators 
        
        # assign other parameters
        self.stdD = stdD #standard deviation of the consumption
        self.UseDataSource = UseDataSource
        self.dispatching_intervals = dispatching_intervals #number of dispatching intervals

        # initialization
        self.pre_reward = 0 #initialize the reward
        self.violation = False #initialize the violation

        # define the upper and lower bounds of the action space
        self.action_space = Box(low = np.full((2*self.NG+2*self.NsG+self.NxG, ), -1), 
                                high = np.full((2*self.NG+2*self.NsG+self.NxG, ), 1), 
                                shape=(2*self.NG+2*self.NsG+self.NxG, ))
        
        # define the upper and lower bounds of the observation space
        self.observation_space = Box(low=np.zeros((2*self.NL, )), 
                                    high=np.full((2*self.NL, ), 1), 
                                    shape=(2*self.NL, ),
                                    dtype=np.float32)
        
        # define the total number of dispatching intervals
        self.episode_length = dispatching_intervals

        if self.UseDataSource == False:
            # assign generation limit, load limit, volatage limit, line limit manually
            self.define_limit_manually()
            # initialize the state with existing single data saved in the grid
            self.state = self.read_consumption_from_grid()
        else:
            # assign limits based on data source profile
            self.define_limit_with_profile()
            # initialize the state with time series
            self.load_profile = self.extract_load_from_sampled_profile()
            self.state = np.append(self.load_profile[0],0) # reactive power is 0 Mvar
        
    

    def step(self, action):
        # intialize the terminated and truncated
        terminated = False
        truncated = False

        # run the power flow
        try:
            pp.runpp(self.net)
            #pp.runpp(self.net, algorithm='gs') 
            control.run_control(self.net, max_iter=30)
        except:
            # output diagnostic report when the power flow does not converge
            print("Power flow does not converge!")
            print(pp.diagnostic(self.net, report_style='detailed')) 
            reward = -5000
            # truncated = True
        else:
            # print("Power flow converges!")
            reward = self.calculate_reward()
            # check if the episode is terminated in the case of convergence and no violation
            if self.violation == False:
                terminated = abs(reward-self.pre_reward) <= 0.001

        # decrease the episode length
        self.episode_length -= 1

        # update the next observation
        if self.UseDataSource == False:
            # random sampling from a normal distr. 
            self.state += self.stdD * (np.random.randn(2*self.NL, ))
        else:
            # update the state with the sampled profile
            self.state = np.append(self.load_profile[(len(self.load_profile) - self.episode_length - 1)], 0) # assume reactive power is 0 MVar

        observation = self._get_observation() # fully observable environment
        # check if the episode is terminated in the case of reaching the end of the episode
        terminated = self.episode_length == 0
        # apply action and state as input to the net
        self.net = self.apply_action_state_to_net(action)
        # print(self.net['converged'])
        info = {}
        # save the reward for the current step
        self.pre_reward = reward
        print("episode length: ", self.episode_length, "Reward:", reward, "; Terminated:", terminated, "; Truncated:", truncated)
        return observation, reward, terminated, truncated, info


    def render(self):
        pass


    def reset(self, seed=None):
        #np.random.seed(seed)
        super().reset(seed=seed)
        # assign initial state of the upcoming episode
        if self.UseDataSource == False:
            # add some noice
            self.state += self.stdD * (np.random.randn(2*self.NL, ))
        else:
            # sampling the startpoint from the load profile
            self.load_profile = self.extract_load_from_sampled_profile()
            self.state = np.append(self.load_profile[0],0) # reactive power is 0 Mvar
        observation = self._get_observation() # fully observable environment
        # reset the episode length
        self.episode_length = self.dispatching_intervals
        info = {}
        print("Reset!")
        return observation, info
    

    def close(self):
        print("Close this episode! The episode length is: ", self.episode_length)

    
    def define_limit_manually(self):
        if 'min_p_mw' not in self.net.gen:
            self.net.gen.loc[:,"min_p_mw"] = self.net.gen.loc[:,"p_mw"] * 0 # assume the min power output is 0 if not specified
        if 'max_p_mw' not in self.net.gen:
            self.net.gen.loc[:,"max_p_mw"] = self.net.gen.loc[:,"p_mw"] * 1.5 # assume the max power output is 1.5 times the current power output if not specified
        self.PGmin = self.net.gen.loc[:,"min_p_mw"]
        self.PGmax = self.net.gen.loc[:,"max_p_mw"]
        self.PsGmin = np.zeros(self.NsG) # assume the min power output from static generator is 0 MW
        self.PsGmax = np.full(self.NsG, 10) 
        self.QsGmin = np.full(self.NsG, -10) # assume the min reactive power output from static generator is -10 MVar
        self.QsGmax = np.full(self.NsG, -10) # assume the max reactive power output from static generator is 10 MVar
        self.VGmin = 0.94
        self.VGmax = 1.06
        self.PLmax = 100
        self.PLmin = 0
        self.QLmax = 100
        self.QLmin = -100
        self.linemax = 100


    def read_consumption_from_grid(self):
        loads = self.net.load.loc[:, ['p_mw', 'q_mvar']].to_numpy() # reads the active and reactive power of the loads
        return loads.flatten("F").astype(np.float32) 

    # load the generation and consumption profile from json file
    def load_json(self, path):
        profile = pd.read_json(path)
        return profile


    def define_limit_with_profile(self):
        profile = self.load_json("/Users/YANG_Chialing/Desktop/Master_Thesis_TUM/pandapower/tutorials/cigre_timeseries_15min.json")
        self.PLmax = profile.loc[:,"residential"].max() * 1.5
        self.PLmin = 0
        self.QLmax = 10
        self.QLmin = 0
        self.PsGmax = np.array([profile.loc[:,"pv"].max(), profile.loc[:,"wind"].max()])
        self.PsGmin = np.zeros(self.NsG)
        self.QsGmax = np.full((self.NsG, ), 1)
        self.QsGmin = np.zeros(self.NsG)
        self.VGmin = 0.94
        self.VGmax = 1.06
        self.linemax = 100
        


    def time_series_sampling(self):
        profile = self.load_json("/Users/YANG_Chialing/Desktop/Master_Thesis_TUM/pandapower/tutorials/cigre_timeseries_15min.json")
        time_steps = profile['residential'].size
        starting_point = random.randint(0, time_steps-1-self.dispatching_intervals)
        sampled_profile = profile.loc[starting_point:starting_point+self.dispatching_intervals-1, :]
        return sampled_profile


    def extract_load_from_sampled_profile(self):
        sampled_profile = self.time_series_sampling()
        # sampled_profile.loc[:, ['pv', 'wind']].to_numpy()
        return sampled_profile.loc[:, ['residential']].to_numpy()



    def normalize(self, value, max_value, min_value, max_space, min_space):
        return (value - min_value) / (max_value - min_value) * (max_space - min_space) + min_space


    def denormalize(self, value, max_value, min_value, max_space, min_space):
        return (value - min_space) / (max_space - min_space) * (max_value - min_value)  + min_value


    def _get_observation(self):
        normalized_obs = []
        for i in range(self.NL):
            normalized_obs.append(self.normalize(self.state[i], self.PLmax, self.PLmin, 1, 0))
        for i in range(self.NL):
            normalized_obs.append(self.normalize(self.state[i+self.NL], self.QLmax, self.QLmin, 1, 0))
        return np.array(normalized_obs).astype(np.float32)


    def apply_action_state_to_net(self, action):
        for i in self.net.gen.index: # PV bus
            self.net.gen.loc[i, 'p_mw'] = self.denormalize(action[i], self.PGmax[i], self.PGmin[i], 1, -1)
            self.net.gen.loc[i, 'vm_pu'] = self.denormalize(action[i+self.NG], self.VGmax, self.VGmin, 1, -1)
        for i in self.net.sgen.index: # PQ bus
            self.net.sgen.loc[i,'p_mw'] = self.denormalize(action[i+2*self.NG], self.PsGmax[i], self.PsGmin[i], 1, -1)
            self.net.sgen.loc[i, 'q_mvar'] = self.denormalize(action[i+2*self.NG+self.NsG], self.QsGmax[i], self.QsGmin[i], 1, -1)
        for i in self.net.ext_grid.index: # slack bus
            self.net.ext_grid.loc[i, 'vm_pu'] = self.denormalize(action[i+2*self.NG+2*self.NsG], self.VGmax, self.VGmin, 1, -1)
        for i in self.net.load.index: # load
            self.net.load.loc[i, 'p_mw'] = self.state[i]#self.denormalize(self.state[i], self.PLmax, self.PLmin, 1, 0)
            self.net.load.loc[i, 'q_mvar'] = self.state[i+self.NL]#self.denormalize(self.state[i+self.NL], self.QLmax, self.QLmin,1 , 0)
        return self.net


    def calculate_reward(self):
        violated_buses = tb.violated_buses(self.net, self.VGmin, self.VGmax) # bus voltage violation 
        violated_lines = tb.overloaded_lines(self.net, self.linemax) #line overload violation
        if violated_buses.size != 0 or violated_lines.size != 0 : #if there is violation
            self.violation = True
            return -500*(len(violated_buses) + len(violated_lines))
        else:
            # return 1000 - 0.1 * self.calculate_gen_cost()
            return 1000- 10*self.calculate_gen_cost()
            

    def calculate_gen_cost(self):
        gen_cost = 0
        if self.net.poly_cost.index.size > 0:
            #if the cost function is polynomial
            for i in self.net.gen.index:
                gen_cost += self.net.res_gen.p_mw[i]**2 * self.net.poly_cost.iat[i,4] + self.net.res_gen.p_mw[i] * self.net.poly_cost.iat[i,3] + self.net.poly_cost.iat[i,2] \
                            + self.net.poly_cost.iat[i,5] + self.net.poly_cost.iat[i,6] * self.net.res_gen.q_mvar[i] + self.net.poly_cost.iat[i,7] * self.net.res_gen.q_mvar[i]**2
            for i in self.net.sgen.index:
                gen_cost += self.net.res_sgen.p_mw[i]**2 * self.net.poly_cost.iat[i,4] + self.net.res_sgen.p_mw[i] * self.net.poly_cost.iat[i,3] + self.net.poly_cost.iat[i,2] \
                            + self.net.poly_cost.iat[i,5] + self.net.poly_cost.iat[i,6] * self.net.res_sgen.q_mvar[i] + self.net.poly_cost.iat[i,7] * self.net.res_sgen.q_mvar[i]**2
            for i in self.net.ext_grid.index:
                gen_cost += self.net.res_ext_grid.p_mw[i]**2 * self.net.poly_cost.iat[i,4] + self.net.res_ext_grid.p_mw[i] * self.net.poly_cost.iat[i,3] + self.net.poly_cost.iat[i,2] \
                            + self.net.poly_cost.iat[i,5] + self.net.poly_cost.iat[i,6] * self.net.res_ext_grid.q_mvar[i] + self.net.poly_cost.iat[i,7] * self.net.res_ext_grid.q_mvar[i]**2
        elif self.net.pwl_cost.index.size > 0:
            #if the cost function is piecewise linear
            points_list = self.net.pwl_cost.at[i, 'points']
            for i in self.net.gen.index:
                for points in points_list:
                    p0, p1, c01 = points
                    if p0 <= self.net.res_gen.p_mw[i] < p1:
                        gen_cost += c01 * self.net.res_gen.p_mw[i]
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
        else:
            #if the cost function is empty
            total_gen = self.net.res_gen['p_mw'].sum() + self.net.res_sgen['p_mw'].sum() + self.net.res_ext_grid['p_mw'].sum()
            gen_cost =  0.1 * total_gen**2 + 40 * total_gen
            # print("No cost function found for generators. Assume the cost function is 0.1 * p_tot**2 + 40 * p_tot.")
        return gen_cost    

def change_gen_into_sgen(grid):
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

def load_test_case_grid(n):
    # load the grid to define the grid topology
    case = f"case{n}" # n: case number in pandapower 
    grid = getattr(pn, case)()
    return grid

def load_simple_grid():
    grid = pn.example_simple()
    grid = change_gen_into_sgen(grid)
    return grid




# Environment Setup
# grid = load_test_case_grid(14)
grid = load_simple_grid()
# env = PowerGridEnv(grid, 0.03, 96, False)
env = PowerGridEnv(grid, 0.03, 24, True) # dispatching interval needs to be smaller than the total data points (96)
#wrapped_env = TimeLimit(env, max_episode_steps)


# train the model
log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", env, n_steps = 100, verbose=1, tensorboard_log=log_path)
# model = PPO("MlpPolicy", env, n_steps = 1280, verbose=1, tensorboard_log=log_path)
# print(model.get_parameters())
model.learn(total_timesteps= 600, progress_bar=True)
# model.learn(total_timesteps= 60000, progress_bar=True)
# model.save("Training/Model/PPO_PowerGrid")

