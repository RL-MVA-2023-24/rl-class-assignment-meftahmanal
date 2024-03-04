import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

# Define the environment
env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)

# Define the DQN network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Replay memory for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), action, reward, np.array(next_state), done

# Agent class implementing the required methods
class ProjectAgent:
    def __init__(self):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = ReplayBuffer(10000)
        self.model = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99

    def act(self, state, use_random=False):
        if random.random() <= self.epsilon or use_random:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state)
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        Q_expected = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        Q_next = self.model(next_states).detach().max(1)[0]
        Q_target = rewards + (self.gamma * Q_next * (1 - dones))
        
        loss = self.criterion(Q_expected, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path='model.pth'):
        torch.save(self.model.state_dict(), path)

    def load(self, path='model.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

# Training loop
def train(agent, episodes=1000, batch_size=64):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.learn(batch_size)
            state = next_state
            total_reward += reward
        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

if __name__ == "__main__":
    agent = ProjectAgent()
    train(agent)
    agent.save('model.pth')
