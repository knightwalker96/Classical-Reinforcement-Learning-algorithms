# Classical Reinforcement Learning Algorithms
This repository contains the implementation of reinforcement learning algorithms as part of an assignment of the course **AIL722: Reinforcement Learning** at IIT Delhi. The assignment explores multiple approaches
like *Policy Iteration, Value Iteration, SARSA, Q-Learning*, and *Deep Q-Networks (DQN)* across custom and benchmark environments.

## Repository Structure

### Part 1
Contains the implementation of Policy Iteration and Value Iteration algorithms on the TreasureHunt environment. **Policy Iteration** iteratively improves a policy by alternating between policy evaluation (computing value functions for a fixed policy) and policy improvement.
**Value Iteration** computes the optimal policy by iteratively updating value functions and deriving the policy directly from the updated values.

### Part 2
Includes the implementation of SARSA and Q-Learning algorithms on both the TreasureHunt and Taxi environments. **SARSA** is an on-policy temporal-difference control method that updates the Q-values based on the current policy and transitions (state-action-reward-state-action).
**Q-Learning** is an off-policy TD control method that updates the Q-values by learning the best possible action for future states, regardless of the agent's current policy.
These algorithms incorporate exploration-exploitation strategies using epsilon-greedy policies and are suited for learning policies in dynamic environments. These algorithms use dynamic programming to find optimal solutions for grid-based environments.

### Part 3
Contains the implementation of a **Deep Q-Network (DQN)** agent on the TreasureHunt-v2 and LunarLander environments. DQN combines Q-Learning with neural networks to approximate Q-values for high-dimensional or continuous state spaces. It employs experience replay and target networks for stable training. DQN is applied to solve complex environments with continuous or very large state spaces and dynamic interactions.

##### AIL722 Assignment 2 is the Problem Statement. 
##### Report.pdf is the detailed report for the assignment, including results and insights. 
##### notebook.ipynb is the Jupyter notebook which contains the step-by-step implementation of all algorithms for better understanding. 

## Results and Inights
The Plots folder in each part contains visualizations such as learning curves, rewards per episode, and convergence behaviors for the respective algorithms.

## How to Use
1. Clone this repository:
   ```bash
   git clone <https://github.com/knightwalker96/Classical-Reinforcement-Learning-algorithms.git>  
   cd <Classical-Reinforcement-Learning-algorithms>
2. Run individual scripts for specific algorithms. Navigate to the corresponding folder (Part 1, Part 2, Part 3) and execute the Python files.
