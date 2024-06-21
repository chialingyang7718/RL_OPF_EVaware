# Exploring Reinforcement Learning Solutions for EV-aware Optimal Power Flow Under Uncertainty
## Project Description
This project explores the use of reinforcement learning (RL) to solve the Optimal Power Flow (OPF) problem in electrical power systems. The goal is to determine the optimal settings of control variables to minimize the cost of power generation while satisfying system constraints. The project is dvided into two phases. At the first phase, a single-period version of OPF is considered. Later on, EV charging/discharging is integrated into the OPF problem, which is transformed into a multi-period OPF. 
### What is Optimal Power Flow (OPF)?
Optimal Power Flow (OPF), typically representing a snapshot of the power system operation, is suitable for short-term operational planning and real-time applications requiring rapid decisions. With the introduction of energy storage and electrical vehicles, MOPF has become popular. It extends the optimization horizon beyond a single time period to consider the dynamic behavior of the power system over multiple time intervals. It accounts for the time-varying nature of electricity demand, renewable energy generation, and the scheduling of energy resources over a longer planning horizon.
### What is Reinforcement Learning (RL)?
Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by performing actions in an environment to achieve largest possible cumulative reward. Unlike supervised learning, where the model is trained on a dataset of correct input-output pairs, RL involves learning through trial and error.
#### Key Concepts in RL
+ Agent: The learner or decision-maker that interacts with the environment.
+ Environment: Everything that the agent interacts with and affects through its actions.
+ State: A representation of the current situation of the environment.
+ Action: A decision made by the agent that affects the state of the environment.
+ Reward: A scalar feedback signal given to the agent after taking an action, indicating the immediate benefit of that action.
+ Policy: A strategy used by the agent to decide which actions to take based on the current state.
#### How RL Works?
1. Initialization: The agent starts with a policy, often random.
2. Interaction: The agent interacts with the environment by observing states, taking actions, and receiving rewards.
3. Learning: Based on the rewards received and the states visited, the agent updates its policy to maximize cumulative rewards.
4. Iteration: This process is repeated over many episodes, allowing the agent to improve its policy through exploration and exploitation.
## File Structure
+ Scripts: This is where all the codes are. It contains
  - CustEnv: customized power grid environment
  - main.py: main script to execute the training of the model for any IEEE test case.
  - main_simple_example.py: main script to train the model for a simple case study.
  - requirements.txt: a script contains all the dependencies.
  - text_env.py: a script to test whether the environment works as it should.
+ Training: This is where the trained log files and models are saved.
+ Literature_Review.xlsx: This contains a table of all the related literatures for this project.

**This Project is still in progress. Therefore, there are still some vital components missing.**
