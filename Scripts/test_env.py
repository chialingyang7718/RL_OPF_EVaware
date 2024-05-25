#import gym
import gymnasium as gym

# import the environment
import CustEnv
# import CustEnv.Env.PowerGridEnv as environment

# import the grid loader
from grid_loader import load_test_case_grid, load_simple_grid, load_simbench_grid

# load the grid for the environment
grid = load_test_case_grid(14)
# grid = load_simple_grid()
# grid_code = "1-HV-urban--0-sw"
# grid = load_simbench_grid(grid_code)


# initiate the environment
# env = environment.PowerGrid(grid, 0.03, 2, False, False)
env_fns = [lambda: gym.make('PowerGrid-v0', net = grid, dispatching_intervals = 2)] * 2
# env = gym.make('PowerGrid-v0', net = grid, dispatching_intervals = 2)
env = gym.vector.AsyncVectorEnv(env_fns, shared_memory=False)



# test the environment by randomly selecting actions
def test_env():
    episodes = 2
    for episode in range(1, episodes+1):
        state = env.reset()
        terminated = False
        truncated = False
        score = 0 
        while terminated == False and truncated == False:
            env.render()
            action = env.action_space.sample()
            n_state, reward, terminated, truncated, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()

if __name__ == '__main__':
    test_env()