from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from interface import Agent
import torch
import torch.nn as nn
import numpy as np
import random
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

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
        self.index = int(self.index)

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.stack(state).to(self.device),
            torch.LongTensor(action).to(self.device),
            torch.FloatTensor(reward).to(self.device),
            torch.stack(next_state).to(self.device),
            torch.FloatTensor(done).to(self.device)
        )

    def __len__(self):
        return len(self.data)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ProjectAgent(Agent):
    def __init__(self, state_size, action_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=100000, device=self.device)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def act(self, state, use_random=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if random.random() < self.epsilon or use_random:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                action_values = self.model(state)
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        Q_expected = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        Q_targets_next = self.model(next_states).detach().max(1)[0]
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        loss = self.loss_fn(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))

def train_agent(agent, env, n_episodes=1000, max_t=200):
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.append(state, action, reward, next_state, float(done))
            state = next_state
            score += reward
            agent.learn()
            if done:
                break
        scores.append(score)
        print(f'Episode {i_episode}\tAverage Score: {np.mean(scores[-100:]):.2f}')
        if i_episode % 100 == 0:
            agent.save(f'checkpoint_episode_{i_episode}.pth')

    return scores

if __name__ == "__main__":
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = ProjectAgent(state_size=state_size, action_size=action_size)
    scores = train_agent(agent, env)

    # Optionally, you can save the final model and scores or plot the scores for analysis.
    agent.save('final_model.pth')

