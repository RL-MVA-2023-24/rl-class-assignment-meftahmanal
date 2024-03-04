import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from env_hiv import HIVPatient
from typing import Protocol

class Agent(Protocol):
    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ProjectAgent:
    def __init__(self, state_size, action_size):
        self.model = SimpleNN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(0, 4)
        with torch.no_grad():
            observation = torch.tensor(observation).float()
            q_values = self.model(observation)
            action = torch.argmax(q_values).item()
        return action

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        try:
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        except FileNotFoundError:
            print(f"No saved model at {path}. Starting from scratch.")

def train():
    env = HIVPatient(clipping=True, logscale=False, domain_randomization=True)
    agent = ProjectAgent(env.observation_space.shape[0], env.action_space.n)
    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # Update agent here based on (state, action, reward, next_state)
            state = next_state
        if episode % 100 == 0:
            print(f"Episode {episode}: completed")
            agent.save("latest_model.pth")

if __name__ == "__main__":
    train()
