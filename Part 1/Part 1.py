import time
import numpy as np
from env import TreasureHunt

def policy_iteration(env, gamma=0.95, max_iterations=100):
     # Start with a uniform policy
    policy = np.ones([env.num_states, env.num_actions]) / env.num_actions
    V = np.zeros(env.num_states)

    for _ in range(max_iterations):
        # Policy evaluation step
        while True:
            delta = 0
            for s in range(env.num_states):
                v = V[s]
                V[s] = sum(policy[s, a] * (sum(env.T[s, a, s_next] * (env.reward[s] + gamma * V[s_next])
                                             for s_next in range(env.num_states)))
                           for a in range(env.num_actions))
                delta = max(delta, abs(v - V[s]))
            if delta < 1e-5:
                break

        # Policy improvement step
        policy_stable = True
        for s in range(env.num_states):
            old_action = np.argmax(policy[s])
            policy[s] = np.eye(env.num_actions)[np.argmax([sum(env.T[s, a, s_next] * (env.reward[s] + gamma * V[s_next])
                                                               for s_next in range(env.num_states))
                                                           for a in range(env.num_actions)])]
            if old_action != np.argmax(policy[s]):
                policy_stable = False
        if policy_stable:
            break

    return policy, V

def value_iteration(env, gamma=0.95, max_iterations=100):
    V = np.zeros(env.num_states)
    policy = np.ones([env.num_states, env.num_actions]) / env.num_actions

    for _ in range(max_iterations):
        delta = 0
        for s in range(env.num_states):
            v = V[s]
            # Compute the expected return for each action and find the best action
            action_values = [sum(env.T[s, a, s_next] * (env.reward[s] + gamma * V[s_next])
                                 for s_next in range(env.num_states))
                             for a in range(env.num_actions)]
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < 1e-5:
            break

    # Derive policy from the value function
    for s in range(env.num_states):
        action_values = [sum(env.T[s, a, s_next] * (env.reward[s] + gamma * V[s_next])
                             for s_next in range(env.num_states))
                         for a in range(env.num_actions)]
        best_action = np.argmax(action_values)
        policy[s] = np.eye(env.num_actions)[best_action]

    return policy, V

if __name__ == "__main__":

    locations = {
        'ship': [(0,0)],
        'land': [(3,0),(3,1),(3,2),(4,2),(4,1),(5,2),(0,7),(0,8),(0,9),(1,7),(1,8),(2,7)],
        'fort': [(9,9)],
        'pirate': [(4,7),(8,5)],
        'treasure': [(4,0),(1,9)]
    }

    env = TreasureHunt(locations)

    # Run Policy Iteration
    start = time.time()
    policy1, V1 = policy_iteration(env)
    env.visualize_policy(policy1, path="policy_iteration_visualization.png")
    env.visualize_policy_execution(policy1, path="policy_trajectory.gif")
    end = time.time()
    print("Time taken: {}".format(end-start))

    # Run Value Iteration
    start = time.time()
    policy2, V2 = value_iteration(env)
    env.visualize_policy(policy2, path="value_iteration_visualization.png")
    env.visualize_policy_execution(policy2, path="value_trajectory.gif")
    end = time.time()
    print("Time taken: {}".format(end-start))
