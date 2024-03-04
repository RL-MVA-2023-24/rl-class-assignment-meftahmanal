import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from env_hiv import HIVPatient

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

class ProjectAgent:
    def __init__(self):
        # Dynamically setting state_size and action_size based on the environment
        env = HIVPatient()
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        self.memory = ReplayBuffer(10000)
        self.policy_net = DQN(self.state_size, self.action_size)
        self.target_net = DQN(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.criterion = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99

    def act(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return np.argmax(self.policy_net(torch.FloatTensor(state)).cpu().data.numpy())
        else:
            return random.randrange(self.action_size)

    def optimize(self, batch_size):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train(agent, episodes, batch_size):
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
            agent.optimize(batch_size)
        if episode % 10 == 0:
            print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {agent.epsilon}")

if __name__ == "__main__":
    env = HIVPatient()
    agent = ProjectAgent()
    train(agent, 1000, 64)
