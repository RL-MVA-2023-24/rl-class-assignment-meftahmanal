import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import deque
import random
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

# Neural Network Model Definition
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)

# Replay Buffer to store experience
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

# Agent Implementation
class ProjectAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(10000)
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_size)
        return action

    def optimize(self, batch_size):
        if len(self.memory) < batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(batch_size)
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_values = self.model(state)
        next_q_values = self.model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + 0.99 * next_q_value * (1 - done)
        
        loss = self.criterion(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path="agent.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="agent.pth"):
        self.model.load_state_dict(torch.load(path))

# Main Training Loop
def train():
    env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = ProjectAgent(state_size, action_size)
    episodes = 1000  # You can adjust this
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 300
    epsilon_by_episode = lambda episode: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * episode / epsilon_decay)

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        while not done:
            epsilon = epsilon_by_episode(episode)
            action = agent.act(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.optimize(64)

        if episode % 10 == 0:
            print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {epsilon}")
            agent.save()

if __name__ == "__main__":
    train()
