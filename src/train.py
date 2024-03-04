from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

class ProjectAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.model = DQN(self.state_dim, self.action_dim).to(device)
        self.target_model = DQN(self.state_dim, self.action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(100000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64

    def act(self, state, use_random=False):
        if random.random() < self.epsilon or use_random:
            return self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state)
            return q_values.max(1)[1].item()

    def train(self, episodes=200):
        for episode in range(episodes):
            state = self.env.reset()[0]
            done = False
            total_reward = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.optimize_model()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {self.epsilon}")
        self.update_target_network()

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(-1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(-1).to(device)

        current_q_values = self.model(states).gather(1, actions)
        max_next_q_values = self.target_model(next_states).max(1)[0].detach().unsqueeze(-1)
        expected_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path="dqn_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="dqn_model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.update_target_network()

env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
agent = ProjectAgent(env)
agent.train(episodes=200)
agent.save("dqn_hiv_patient.pth")
