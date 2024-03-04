import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import namedtuple, deque
import random
from env_hiv import HIVPatient  

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.network(x)

class ProjectAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = SimpleNN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        
    def act(self, observation, use_random=False):
        if use_random or random.random() < 0.1:  
            return random.choice(range(self.action_size))
        self.model.eval()
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
            action_probs = self.model(observation)
            action = torch.argmax(action_probs).item()
        return action
    
    def save(self, path='agent.pth'):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path='agent.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

def train_agent(episodes=1000, save_path='agent.pth'):
    env = HIVPatient()
    agent = ProjectAgent(env.observation_space.shape[0], env.action_space.n)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            agent.optimizer.zero_grad()
            target = torch.tensor([action], dtype=torch.long)
            prediction = agent.model(torch.tensor(state, dtype=torch.float).unsqueeze(0))
            loss = agent.criterion(prediction, target)
            loss.backward()
            agent.optimizer.step()
            
            state = next_state
        
        if episode % 100 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}')
            agent.save(save_path)
    
    agent.save(save_path)

if __name__ == '__main__':
    train_agent()
