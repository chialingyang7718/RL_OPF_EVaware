# Exploring Reinforcement Learning Solutions for EV-aware Optimal Power Flow under Uncertainty
## Project Description
This project explores whether reinforcement learning (RL), specifically Proximal Policy Optimization (PPO), can be an effective approach for solving EV-aware Optimal Power Flow (OPF) problems under uncertainty. Furthermore, a conventional optimization approach, Interior Point Method (IPM) is served as a benchmark for comparison.
### What is Optimal Power Flow (OPF)?
Optimal Power Flow (OPF), typically representing a snapshot of the power system operation, is suitable for short-term operational planning and real-time applications requiring rapid decisions. With the introduction of energy storage and EVs, MOPF has become popular. It extends the optimization horizon beyond a single time period to consider the dynamic behavior of the power system over multiple time intervals. It accounts for the time-varying nature of electricity demand, renewable energy generation, and the scheduling of energy resources over a longer planning horizon.
### What is Reinforcement Learning (RL)?
Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by performing actions in an environment to achieve largest possible cumulative reward. Unlike supervised learning, where the model is trained on a dataset of correct input-output pairs, RL involves learning through trial and error.
#### Key Concepts in RL
+ Agent: The learner or decision-maker that interacts with the environment. Here, the Distribution System Operator represnets the Agent.
+ Environment: Everything that the agent interacts with and affects through its actions. In this project, the Environment is represented by the power grid system.
+ State: A representation of the current situation of the environment. In this project, the states are active power of load, reactive power of laod, EV state of charge(SOC) and EV connection status.
+ Action: A decision made by the agent that affects the state of the environment. Here, the actions are active power of generation, voltage output of generation and EV charging active power.
+ Reward: A scalar feedback signal given to the agent after taking an action, indicating the immediate benefit of that action.
+ Policy: A strategy used by the agent to decide which actions to take based on the current state.
+ Algorithm: There are many state-of-art RL algorithms. PPO is selected for this project.
#### How RL Works?
1. Initialization: The agent starts with a policy, often random.
2. Interaction: The agent interacts with the environment by observing states, taking actions, and receiving rewards.
3. Learning: Based on the rewards received and the states visited, the agent updates its policy to maximize cumulative rewards.
4. Iteration: This process is repeated over many episodes, allowing the agent to improve its policy through exploration and exploitation.
## File Structure
+ Scripts: This is where all the codes are. It contains
  - CustEnv: customized power grid environment
  - grid_loader.py: a module to load grid from different sources.
  - main_train.py: main script to execute the training of the model for any IEEE test case.
  - evaluate.py: to evaluate PPO model and generate the data input for IPM.
  - IPM.py: main script to solve EV-aware OPF with IPM.
  - requirements.txt: a txt file contains all the dependencies.
  - data_visualization.ipynb: a jupyter notebook to conduct data analysis and generate visualization for the results.
+ Training/Model: This is where the trained models are saved.
+ Evaluation: This is where the output of evaluate.py stores.
+ Literature_Review.xlsx: This contains a table of all the related literatures for this project.
## EV Profile Data Source: https://zenodo.org/records/4514928

