import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
    def __init__(self, state_size, action_size, load_model=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(10000)
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.epsilon = 0.1
        if load_model:
            self.load('model.pth')

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            return q_values.max(1)[1].item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        if os.path.isfile(path):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
        else:
            print(f"No model file found at {path}. Starting from scratch.")

def train(agent, episodes=1000):
    env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
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
        if episode % 100 == 0:
            print(f"Episode {episode}, Total reward: {total_reward}")
            agent.save('model.pth')

if __name__ == "__main__":
    env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = ProjectAgent(state_size, action_size, load_model=True)
    train(agent)
