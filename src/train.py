from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random

env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

class ProjectAgent:
    def __init__(self):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = ReplayBuffer(10000)
        self.model = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.epsilon = 0.1

    def act(self, state, use_random=False):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state)
            return action_values.max(1)[1].item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path="model.pth"):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

def train(agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        print(f"Episode: {episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    agent = ProjectAgent()
    train(agent, 1000)
    agent.save("model.pth")
