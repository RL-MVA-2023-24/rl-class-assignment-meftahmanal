import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from env_hiv import HIVPatient
from gymnasium.wrappers import TimeLimit

# DQN Network Architecture
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Replay Buffer for storing experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ProjectAgent using DQN
class ProjectAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4, gamma=0.99, buffer_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = learning_rate
        self.model = DQNNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.criterion = nn.MSELoss()

    def act(self, observation, use_random=False):
        if use_random or random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        else:
            observation = torch.FloatTensor(observation).unsqueeze(0)
            q_values = self.model(observation)
            return q_values.max(1)[1].item()

    def experience_replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = [np.array(a) for a in zip(*transitions)]
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)

        current_q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = self.criterion(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

# Training Function
def train(agent, episodes=1000):
    env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.experience_replay()
            state = next_state
            total_reward += reward
        print(f'Episode {episode+1}: Total Reward = {total_reward}')

if __name__ == "__main__":
    env = HIVPatient()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ProjectAgent(state_dim, action_dim)
    train(agent, 1000)
