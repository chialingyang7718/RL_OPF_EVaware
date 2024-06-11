#import gym
import gymnasium as gym

# import the environment
import CustEnv


# import the grid loader
from grid_loader import load_test_case_grid, load_simple_grid, load_simbench_grid

# load the grid for the environment
grid = load_test_case_grid(14)
# grid = load_simple_grid()
# grid_code = "1-HV-urban--0-sw"
# grid = load_simbench_grid(grid_code)

if __name__ == '__main__':
    # initiate the environment for vectorizing
    env_fns = [lambda: gym.make('PowerGrid-v0', net = grid, dispatching_intervals = 2)] * 2
    # vectorize the environment
    env = gym.vector.AsyncVectorEnv(env_fns, shared_memory=True)
    terminated = False
    truncated = False

    # test the environment by randomly selecting actions
    # def test_env():
    episodes = 2
    for episode in range(1, episodes+1):
        state = env.reset()
        score = 0
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)
        while terminated.any() == False and truncated.any() == False:
            action = env.action_space.sample()
            n_state, reward, terminated, truncated, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()


