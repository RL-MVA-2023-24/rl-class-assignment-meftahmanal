from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from interface import Agent
import torch
import torch.nn as nn
import numpy as np
import random
import os

# Wrap the HIV environment to limit the number of steps per episode
env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states).to(self.device),
            torch.tensor(actions, device=self.device, dtype=torch.long),
            torch.tensor(rewards, device=self.device, dtype=torch.float),
            torch.stack(next_states).to(self.device),
            torch.tensor(dones, device=self.device, dtype=torch.float)
        )

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ProjectAgent(Agent):
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.model = QNetwork(state_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(10000, device)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def act(self, state):
        state = torch.tensor([state], device=self.device, dtype=torch.float)
        if random.random() > self.epsilon:
            with torch.no_grad():
                action_values = self.model(state)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.append(state, action, reward, next_state, done)
        self.learn()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        Q_targets_next = self.model(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.model(states).gather(1, actions.unsqueeze(1))

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(n_episodes=2000, max_t=1000):
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)
        print(f'Episode {i_episode}\tScore: {score:.2f}')
    return scores

# Define state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = ProjectAgent(state_size, action_size, device='cuda' if torch.cuda.is_available() else 'cpu')
scores = train_agent()

# Save the trained model
torch.save(agent.model.state_dict(), 'model.pth')
