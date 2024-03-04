from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from agent import DQNAgent
import torch
from config import config

env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)

class ProjectAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.agent = DQNAgent(self.state_size, self.action_size, config)

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()[0]
            total_reward = 0
            done = False
            while not done:
                action = self.agent.act(state, config['epsilon_max'])
                next_state, reward, done, _, _ = self.env.step(action)
                self.agent.memory.append(state, action, reward, next_state, done)
                self.agent.optimize_model()
                state = next_state
                total_reward += reward
            print(f"Episode {episode+1}, Total Reward: {total_reward}")

    def save(self, path):
        self.agent.save(path)

    def load(self, path):
        self.agent.load(path)

if __name__ == "__main__":
    project_agent = ProjectAgent(env)
    project_agent.train(episodes=100) 
    project_agent.save('final_model.pth')
