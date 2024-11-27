import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import gymnasium as gym
import tqdm

# Action selection function (ε-greedy)
def choose_action(state, Q, epsilon, NActions):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(NActions))
    else:
        return torch.argmax(Q[state]).item()

# Extract policy from the Q-table
def extract_policy(Q, NActions):
    policy = torch.zeros(Q.shape[0], dtype=torch.long)
    for state in range(Q.shape[0]):
        policy[state] = torch.argmax(Q[state])
    return policy

# Epsilon decay function for Taxi-v3
def epsilon_decay_function(epsilon_start, decay_factor, episode, K, epsilon_end=0.1):
    return max(epsilon_start * (decay_factor ** (episode / K)), epsilon_end)

def plot_metrics(rewards, epsilons, title):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Rewards', color='tab:blue')
    ax1.plot(rewards, color='tab:blue', label="Rewards")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon', color='tab:red')
    ax2.plot(epsilons, color='tab:red', label="Epsilon")
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(title)
    fig.tight_layout()
    plt.show()

def sarsa(env, gamma=0.95, alpha=0.05, epsilon=0.4, MaxEpisodes=20000, epsilon_decay=None):
    NStates = env.observation_space.n
    NActions = env.action_space.n

    Q = torch.zeros([NStates, NActions])
    rewards_per_episode = []
    epsilons_per_episode = []
    stable_policy_count = 0
    convergence_episodes = 500
    max_steps = 500

    prev_policy = extract_policy(Q, NActions)

    for episode in tqdm(range(MaxEpisodes)):
        state = env.reset()  # Initial state
        state = 0
        action = choose_action(state, Q, epsilon, NActions)
        total_reward = 0

        done = False
        steps = 0
        while not done:
            next_state, reward,done,_,_ = env.step(action)
            next_action = choose_action(next_state, Q, epsilon, NActions)
            total_reward += reward
            steps += 1
            if steps >= max_steps:
                break

            # SARSA Q-value update rule
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            state = next_state
            action = next_action

        rewards_per_episode.append(total_reward)
        epsilons_per_episode.append(epsilon)

        if epsilon_decay:
            epsilon = epsilon_decay(epsilon, episode)

        #policy = extract_policy(Q, NActions)
        current_policy = extract_policy(Q, NActions)
        if torch.equal(current_policy , prev_policy):
            stable_policy_count += 1
        else:
            stable_policy_count = 0

        prev_policy = current_policy

        if stable_policy_count >= convergence_episodes:
            break

    policy = extract_policy(Q, NActions)
    return Q, policy, rewards_per_episode, epsilons_per_episode

def q_learning(env, gamma=0.95, alpha=0.05, epsilon=0.4, MaxEpisodes=20000, epsilon_decay=None):
    NStates = env.observation_space.n
    NActions = env.action_space.n

    Q = torch.zeros([NStates, NActions])
    rewards_per_episode = []
    epsilons_per_episode = []
    stable_policy_count = 0
    convergence_episodes = 500
    max_steps = 500

    prev_policy = extract_policy(Q, NActions)

    for episode in tqdm(range(MaxEpisodes)):
        state = env.reset()[0]  # Initial state
        total_reward = 0

        done = False
        steps = 0
        while not done:
            action = choose_action(state, Q, epsilon, NActions)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            if steps >= max_steps:
                break

            # Q-Learning Q-value update rule
            Q[state, action] += alpha * (reward + gamma * torch.max(Q[next_state]) - Q[state, action])

            state = next_state

        rewards_per_episode.append(total_reward)
        epsilons_per_episode.append(epsilon)

        if epsilon_decay:
            epsilon = epsilon_decay(epsilon, episode)

    current_policy = extract_policy(Q, NActions)
    if torch.equal(current_policy , prev_policy):
        stable_policy_count += 1
    else:
        stable_policy_count = 0

    prev_policy = current_policy

    #if stable_policy_count >= convergence_episodes:
     #   break

    policy = extract_policy(Q, NActions)
    return Q, policy, rewards_per_episode, epsilons_per_episode

def evaluate_policy(env, q_table, display_episodes=5, total_episodes=100, max_steps = 500):
    """Evaluate the learned Q-table policy over 100 episodes and report the mean reward."""
    total_reward = 0
    total_steps = 0
    rewards_per_episode = []

    for i_episode in range(total_episodes):
        state, info = env.reset()
        rewards, steps = 0, 0

        done = False
        while not done:
            action = np.argmax(q_table[state]).item()
            state, reward,done,_,_ = env.step(action)
            steps += 1
            rewards += reward

            if steps > max_steps:
                break

        rewards_per_episode.append(rewards)
        total_reward += rewards

    # Compute average reward per episode
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

if __name__ == "__main__":

    env2 = gym.make('Taxi-v3')

    # Epsilon decay parameters
    epsilon_start = 0.4
    decay_factor = 0.95
    K = 500

    # Train SARSA on Taxi-v3 (with decaying epsilon)
    sarsa_q_table2, sarsa_policy2, sarsa_rewards2, sarsa_epsilons2 = sarsa(
        env2, gamma=0.95, alpha=0.05, epsilon=epsilon_start, MaxEpisodes=50000,
        epsilon_decay=lambda eps, episode: epsilon_decay_function(epsilon_start, decay_factor, episode, K)
    )
    torch.save(sarsa_q_table2, "sarsa_q_table_taxi.pt")
    torch.save(sarsa_policy2, "sarsa_policy_taxi.pt")
    plot_metrics(sarsa_rewards2, sarsa_epsilons2, "Rewards vs Episodes (SARSA on Taxi-v3)")

    # Train Q-Learning on Taxi-v3 (with decaying epsilon)
    q_learning_q_table2, q_learning_policy2, q_learning_rewards2, q_learning_epsilons2 = q_learning(
        env2, gamma=0.95, alpha=0.05, epsilon=epsilon_start, MaxEpisodes=50000,
        epsilon_decay=lambda eps, episode: epsilon_decay_function(epsilon_start, decay_factor, episode, K)
    )
    torch.save(q_learning_q_table2, "q_learning_q_table_taxi.pt")
    torch.save(q_learning_policy2, "q_learning_policy_taxi.pt")
    plot_metrics(q_learning_rewards2, q_learning_epsilons2, "Rewards vs Episodes (Q-Learning on Taxi-v3)")
