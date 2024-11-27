import math
import sys
from typing import Optional
import numpy as np
import pygame
from pygame import gfxdraw
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
import moviepy.editor as mpy

"""import Box2D
from Box2D.b2 import (
    edgeShape,
    circleShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    contactListener,
)"""

import gym
from gym import error, spaces
from gym.utils import seeding, EzPickle

FPS = 120
SCALE = 60.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400


"""class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False"""


class LunarLander(gym.Env, EzPickle):


    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": FPS}

    def __init__(self, continuous: bool = False):
        EzPickle.__init__(self)
        self.screen = None
        self.isopen = True
        self.gravity = np.array([0.0, -9.8])
        self.lander = None
        self.particles = []
        self.continuous = continuous

        # Observation and action spaces remain the same
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        if self.continuous:
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4)

    def _destroy(self):
        """Simplified destroy function."""
        self.lander = None
        self.particles = []


    def reset(self, seed = None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        self._destroy()
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Helipad position (define the x and y coordinates for the helipad)
        self.helipad_x1 = VIEWPORT_W / 2 - W / 6
        self.helipad_x2 = VIEWPORT_W / 2 + W / 6
        self.helipad_y = H / 4  # You can adjust this based on your desired helipad height

        # Initialize lander position
        initial_y = VIEWPORT_H / SCALE
        self.lander = {
            "position": np.array([VIEWPORT_W / 2, initial_y]),
            "velocity": np.array([0.0, 0.0]),
            "angle": 0.0,
            "angular_velocity": 0.0,
            "fuel": 100.0,
            "ground_contact": [False, False],
        }

        # Random forces to add a bit of complexity
        self.lander["velocity"] += self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM, size=2) / SCALE
        self.lander["angular_velocity"] += self.np_random.uniform(-1.0, 1.0)

        # Initialize sky polygons (simple rectangle for now)
        self.sky_polys = [
            [(0, 0), (W, 0), (W, H), (0, H)]  # Simple full background polygon
        ]

        self.drawlist = [self.lander]  # Add lander to drawlist
        self.legs = [{"position": np.array([0.0, 0.0]), "ground_contact": False},  # Example legs
                     {"position": np.array([0.0, 0.0]), "ground_contact": False}]

        self.drawlist += self.legs  # Add legs to the drawlist

        return np.array([0.0] * 8, dtype=np.float32)  # Simplified return for now


    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                #shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                shape = pygame.draw.circle(screen, (0, 0, 255), position.astype(int), radius),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        # Ensure action is within bounds
        if self.continuous:
            action = np.clip(action, -1, 1)
        else:
            assert self.action_space.contains(action), "Invalid action"

        # Physics update (simplified)
        main_engine_force = 0.0
        side_engine_force = 0.0

        # Main engine control
        if (self.continuous and action > 0) or (not self.continuous and action == 2):
            main_engine_force = MAIN_ENGINE_POWER * action if self.continuous else MAIN_ENGINE_POWER
            self.lander["fuel"] -= main_engine_force * 0.1

        # Side engines
        if (self.continuous and abs(action) > 0.5) or (not self.continuous and action in [1, 3]):
            side_engine_force = SIDE_ENGINE_POWER * action if self.continuous else SIDE_ENGINE_POWER * (action - 2)
            self.lander["fuel"] -= abs(side_engine_force) * 0.1

        # Update velocity and position (simple physics)
        self.lander["velocity"] += self.gravity / FPS + np.array([side_engine_force, main_engine_force]) / FPS
        self.lander["position"] += self.lander["velocity"] / FPS
        self.lander["angle"] += self.lander["angular_velocity"] / FPS

        # Check for ground contact (basic collision with the ground)
        if self.lander["position"][1] <= 0:
            self.lander["position"][1] = 0
            self.lander["velocity"][1] = 0
            self.lander["ground_contact"] = [True, True]

        done = False
        reward = -1  # Penalize each step
        if self.lander["ground_contact"][0] and self.lander["ground_contact"][1]:
            done = True
            reward += 100  # Reward for successful landing

        return np.array([0.0] * 8, dtype=np.float32), reward, done, {}


    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))

        self.surf = pygame.Surface(self.screen.get_size())

        #pygame.transform.scale(self.surf, (SCALE, SCALE))
        #pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())
        self.surf = pygame.transform.scale(self.surf, (VIEWPORT_W, VIEWPORT_H))

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )

        self._clean_particles(False)

        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (50, 50, 50), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (50, 50, 50))

        # Draw the objects in the drawlist (e.g., lander, legs)
        for obj in self.drawlist:
            # Assuming obj has 'position' and 'angle' attributes
            position = obj["position"]
            angle = obj["angle"] if "angle" in obj else 0.0  # Handle objects without angle
            color = (0, 0, 255)  # Example color for lander and legs

            # Example drawing logic for lander and legs (can be customized as needed)
            pygame.draw.circle(
                self.surf, color, (int(position[0] * SCALE), int(position[1] * SCALE)), 10
            )

        for x in [self.helipad_x1, self.helipad_x2]:

            x = x * SCALE
            flagy1 = self.helipad_y * SCALE
            flagy2 = flagy1 + 50
            pygame.draw.line(
            self.surf,
            color=(255, 255, 255),
            start_pos=(x, flagy1),
            end_pos=(x, flagy2),
            width=1,
            )
            pygame.draw.polygon(
            self.surf,
            color=(204, 204, 0),
            points=[
            (x, flagy2),
            (x, flagy2 - 10),
            (x + 25, flagy2 - 5),
            ],
            )
            gfxdraw.aapolygon(
            self.surf,
            [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
            (204, 204, 0),
            )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if mode == "human":
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False


def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
                  s[6] 1 if first leg has contact, else 0
                  s[7] 1 if second leg has contact, else 0
    returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


def demo_heuristic_lander(env, seed=None, render=False):
    total_reward = 0
    steps = 0
    s = env.reset(seed=seed)
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False:
                break

        if steps % 20 == 0 or done:
            print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if done:
            break
    if render:
        env.close()
    return total_reward

class QNetwork(nn.Module):
    """Define Neural Network Architecture for Q-Learning"""

    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Map state to action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size, device):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class Agent():
    """Interacts with and learns from the environment using DQN."""

    def __init__(self, state_size, action_size, seed, buffer_size, batch_size, gamma, tau, lr, update_every, device):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        self.batch_size = batch_size

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory and use random sample from buffer to learn."""
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size, self.device)
                self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using a batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values from target model for next states
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def dqn(agent, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, decay_factor = 0.85, K = 100):
    """Deep Q-Learning."""
    scores = []
    eps = eps_start
    ct = 0
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        frames = []
        for t in range(max_t):
            #frame = env.render(mode = 'human')
            #frames.append(frame)
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                #print("Terminated at step {}".format(t+1))
                ct += 1
                break
        scores.append(score)
        eps = epsilon_decay_function(eps_start, decay_factor, i_episode, K, eps_end)
        k = 100
        if i_episode % k == 0:
            print(f"Episode {i_episode}/{n_episodes}, Score: {np.mean(scores[-k:])}, Epsilon: {eps}")
            #frame_rate = 90  # Set the frame rate for the video
            #clip = mpy.ImageSequenceClip(frames, fps=frame_rate)
            #clip.write_videofile(f"Lunar_lander_trajectory_{i_episode}.mp4", codec="libx264")
    #print("Terminated on own: {}".format(ct))

    return scores

def epsilon_decay_function(epsilon_start, decay_factor, episode, K, epsilon_end=0.1):
    return max(epsilon_start * (decay_factor ** (episode / K)), epsilon_end)

def random_Agent(env, n_episodes=100, max_t=1000):
    """Random Agent on Lunar Lander"""
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        frames = []
        for t in range(max_t):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        scores.append(score)

    return scores

if __name__ == "__main__":

    # Initialize the environment and agent
    env = LunarLander()
    #env = gym.make('LunarLander-v2', continuous=False)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(state_size=state_size, action_size=action_size, seed=0, buffer_size=100000, batch_size=64, gamma=0.99, tau=1e-3, lr=5e-4, update_every=4, device=device)

    # Train the agent
    scores = dqn(agent, env, n_episodes=500, eps_start=1.0, eps_end=0.005, decay_factor = 0.9, K = 100)
    print(sum(scores[-50:]) / 50)

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    random_agent_scores = random_Agent(env)
    print(random_agent_scores)
    print("The average score is: {}".format(sum(random_agent_scores) / len(random_agent_scores)))

    plt.plot(np.arange(len(random_agent_scores)), random_agent_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
