# import gym
import gymnasium as gym

# import the environment
import CustEnv


# import the grid loader
from grid_loader import load_test_case_grid

# load the grid for the environment
grid = load_test_case_grid(5)

# EVScenarios = ["ImmediateFull", "ImmediateBalanced", "Home", "Night"]
EVScenarios = ["ImmediateBalanced"]
if __name__ == "__main__":
    # initiate the environment for vectorizing
    # for EVScenario in EVScenarios:
    env_fns = [
        lambda: gym.make(
            "PowerGrid-v2",
            net=grid,
            dispatching_intervals=2,
            # EVScenario = EVScenario, #Options: ImmediateFull, ImmediateBalanced, Home, Night
            Training=True,
        )
    ] * 2
    # vectorize the environment
    env = gym.vector.AsyncVectorEnv(env_fns, shared_memory=True)
    terminated = False
    truncated = False
    # test the environment by randomly selecting actions
    episodes = 10
    for episode in range(1, episodes + 1):
        state = env.reset()
        score = 0
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)

        while terminated.any() == False and truncated.any() == False:
            action = env.action_space.sample()
            n_state, reward, terminated, truncated, info = env.step(action)
            score += reward
        print("Episode:{} Score:{}".format(episode, score))
    env.close()
