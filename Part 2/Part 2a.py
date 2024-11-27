import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import moviepy.editor as mpy
import tqdm
from env import TreasureHunt

def choose_action(state, Q, epsilon, n_actions):
    #Choose action using epsilon-greedy policy.
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    return np.argmax(Q[state])

def sarsa(env, gamma=0.95, alpha=0.3, epsilon=0.4, max_episodes=1000):
    n_states = env.num_states
    n_actions = env.num_actions
    Q = np.zeros((n_states, n_actions))
    rewards_per_episode = []
    stable_count = 0
    convergence_episodes = 5

    #previous_policy = np.argmax(Q , axis = 1)

    for episode in tqdm(range(max_episodes)):
        state, _ = env.reset()
        action = choose_action(state, Q, epsilon, n_actions)
        Q_old = np.copy(Q)
        total_reward = 0

        i=0
        while True:
            next_state, reward = env.step(action)
            next_action = choose_action(next_state, Q, epsilon, n_actions)

            # SARSA update rule
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            state = next_state
            action = next_action
            total_reward += reward

            if state == env.n * env.n - 1:
                #print(f"Reached goal state at episode {episode}, step {i}, reward till then {total_reward}")
                break
            i += 1

        rewards_per_episode.append(total_reward)

        #current_policy = np.argmax(Q, axis=1)

        max_q_change = np.abs(Q - Q_old).max()
        if max_q_change < 1e-3:
            stable_count += 1
        else:
            stable_count = 0

        """if np.array_equal(current_policy, previous_policy):
            stable_policy_count += 1
        else:
            stable_policy_count = 0  # Reset if the policy changes

        previous_policy = current_policy"""

        if stable_count >= convergence_episodes:
            print(f"Policy converged after {episode} episodes.")
            break

    policy = np.argmax(Q, axis=1)
    return Q, policy, rewards_per_episode

def q_learning(env, gamma=0.95, alpha=0.3, epsilon=0.4, max_episodes=1000):
    n_states = env.num_states
    n_actions = env.num_actions
    Q = np.zeros((n_states, n_actions))
    rewards_per_episode = []
    stable_count = 0
    convergence_episodes = 5

    #previous_policy = np.argmax(Q , axis = 1)

    for episode in tqdm(range(max_episodes)):
        state, _ = env.reset()
        total_reward = 0
        Q_old = np.copy(Q)
        i=0
        while True:
            action = choose_action(state, Q, epsilon, n_actions)
            next_state, reward = env.step(action)

            # Q-learning update rule
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            state = next_state
            total_reward += reward

            if state == env.n * env.n - 1:
                break
            i += 1

        rewards_per_episode.append(total_reward)

        #current_policy = np.argmax(Q, axis=1)

        max_q_change = np.abs(Q - Q_old).max()
        if max_q_change < 1e-3:
            stable_count += 1
        else:
            stable_count = 0

        """if np.array_equal(current_policy, previous_policy):
            stable_policy_count += 1
        else:
            stable_policy_count = 0  # Reset if the policy changes

        previous_policy = current_policy"""

        if stable__count >= convergence_episodes:
            print(f"Policy converged after {episode} episodes.")
            break

    policy = np.argmax(Q, axis=1)
    return Q, policy, rewards_per_episode

    if __name__ == "__main__":

        locations = {
            'ship': [(0,0)],
            'land': [(3,0),(3,1),(3,2),(4,2),(4,1),(5,2),(0,7),(0,8),(0,9),(1,7),(1,8),(2,7)],
            'fort': [(9,9)],
            'pirate': [(4,7),(8,5)],
            'treasure': [(4,0),(1,9)]
        }
        # Assuming you have the TreasureHunt environment ready
        env = TreasureHunt(locations)
        # Train SARSA
        sarsa_q_table, sarsa_policy, sarsa_rewards= sarsa(env, gamma=0.95, alpha=0.3, epsilon=0.4, max_episodes=50000)
        torch.save(sarsa_q_table, "sarsa_q_table_treasure.pt")
        torch.save(sarsa_policy, "sarsa_policy_treasure.pt")

        plt.plot(sarsa_rewards, label='SARSA')
        plt.plot(q_rewards, label='Q-Learning')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Reward vs Episodes for SARSA and Q-Learning')
        plt.legend()
        plt.show()

        def evaluate_policy(env, q_table, display_episodes=5, total_episodes=100, max_steps = 500):
            total_reward = 0
            total_steps = 0
            rewards_per_episode = []

            for i_episode in range(total_episodes):
                state, info = env.reset()
                rewards, steps = 0, 0

                while True:
                    action = np.argmax(q_table[state]).item()
                    state, reward = env.step(action)
                    steps += 1
                    rewards += reward

                    if state == env.n * env.n - 1:
                        break

                    if steps > max_steps:
                        break

                rewards_per_episode.append(rewards)
                total_reward += rewards

            avg_reward = total_reward / total_episodes

            print(f"Results after {total_episodes} episodes:")
            print(f"Average reward per episode: {avg_reward}")

            # Plot the rewards per episode
            plt.plot(rewards_per_episode)
            plt.xlabel('Episodes')
            plt.ylabel('Total Reward')
            plt.title(f'Reward per Episode over {total_episodes} Episodes')
            plt.show()

            return avg_reward

        # Assuming the environment and Q-table are already defined
        q_table_1 = torch.load("q_learning_q_table_treasure.pt")
        q_table_2 = torch.load("sarsa_q_table_treasure.pt")

        # Evaluate the Q-learning policy
        print("For SARSA:")
        avg_reward_sarsa = evaluate_policy(env, q_table_2, display_episodes=5, total_episodes=100, max_steps=500)
        print("For Q-Learning")
        avg_reward_qlearning = evaluate_policy(env, q_table_1, display_episodes=5, total_episodes=100, max_steps=500)

        sarsa_policy = torch.load("sarsa_policy_treasure.pt")
        sarsa_policy = np.array(sarsa_policy)
        num_states = len(sarsa_policy)
        num_actions = 4
        policy = np.zeros((num_states , num_actions))
        for s in range(num_states):
            action = sarsa_policy[s]
            policy[s, action] = 1.0
        env.visualize_policy(policy, path="sarsa_treasure.png")
        env.visualize_policy_execution(policy, path="sarsa_policy_treasure.gif")

        q_policy = torch.load("q_learning_policy_treasure.pt")
        q_policy = np.array(q_policy)
        num_states = len(sarsa_policy)
        num_actions = 4
        policy = np.zeros((num_states , num_actions))
        for s in range(num_states):
            action = sarsa_policy[s]
            policy[s, action] = 1.0
        env.visualize_policy(policy, path="q_treasure.png")
        env.visualize_policy_execution(policy, path="q_policy_treasure.gif")
