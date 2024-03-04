import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from interface import Agent

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 256),  # Assuming 6 observation variables
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)   # Assuming 4 actions
        )
    
    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class ProjectAgent():
    def __init__(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is not trained
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99  # Discount factor
    
    def act(self, state, use_random=False):
        if use_random or np.random.rand() <= self.epsilon:
            return np.random.choice(4)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.policy_net(state)
        return action_values.max(1)[1].item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        states, actions, rewards, next_states, dones = [np.array(lst) for lst in batch]
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q = self.target_net(next_states).max(1)[0]
        expected_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train(env, agent, episodes=1000):
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state, use_random=True)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
        if episode % 10 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {agent.epsilon}")
        
if __name__ == "__main__":
    env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
    agent = ProjectAgent()
    train(env, agent)
    agent.save("ddqn_agent.pth")
