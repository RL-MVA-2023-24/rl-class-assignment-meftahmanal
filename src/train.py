from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from tqdm import tqdm

env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer_sequence = nn.Sequential(
            nn.Linear(input_size,50),
            nn.SELU(),
            nn.Linear(128, 50),
            nn.SELU(),
            nn.Linear(50, output_size)
        )
    
    def forward(self, x):
        return self.layer_sequence(x)

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_space, action_space, config):
        self.state_space = state_space
        self.action_space = action_space
        self.config = config
        
        self.policy_net = NeuralNetwork(state_space, action_space).to(device)
        self.target_net = NeuralNetwork(state_space, action_space).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        self.memory = ExperienceReplayBuffer(config['buffer_size'])
        self.steps_done = 0
        
    def select_action(self, state, epsilon_threshold):
        sample = random.random()
        if sample > epsilon_threshold:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_space)]], device=device, dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.config['batch_size']:
            return
        transitions = self.memory.sample(self.config['batch_size'])
        batch = list(zip(*transitions))

        # Process the batch of experiences
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_state_batch = torch.cat(batch[3])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.config['gamma']) + reward_batch

        # Compute loss
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

class ProjectAgent:
    def __init__(self, env, config):
        self.env = env
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n
        self.agent = DQNAgent(input_size, output_size, config)
        self.config = config

    def train(self):
        episode_rewards = []
        epsilon = self.config['epsilon_max']
        for episode in tqdm(range(self.config['max_episodes']), desc="Training"):
            state = self.env.reset()[0]
            state = np.array(state, dtype=np.float32)
            total_reward = 0
            done = False
            while not done:
                epsilon_threshold = epsilon - episode / self.config['epsilon_decay_period']
                action = self.agent.select_action(state, max(self.config['epsilon_min'], epsilon_threshold))
                next_state, reward, done, _, _ = self.env.step(action.item())
                next_state = np.array(next_state, dtype=np.float32)
                
                reward = torch.tensor([reward], device=device)
                action = torch.tensor([[action]], device=device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                done_tensor = torch.tensor([done], device=device, dtype=torch.float32)

                self.agent.memory.add((state_tensor, action, reward, next_state_tensor, done_tensor))

                self.agent.optimize_model()
                state = next_state
                total_reward += reward.item()
            
            episode_rewards.append(total_reward)
            if episode % self.config['update_target_freq'] == 0:
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

            if episode % self.config['model_save_freq'] == 0:
                self.save(f'model_episode_{episode}.pth')
                
            print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {max(self.config['epsilon_min'], epsilon_threshold)}")
        
        print("Training complete.")
        return episode_rewards

    def save(self, filename):
        torch.save(self.agent.policy_net.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        self.agent.policy_net.load_state_dict(torch.load(filename, map_location=device))
        self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
        print(f"Model loaded from {filename}")

config = {
    'nb_actions': env.action_space.n,
    'learning_rate': 0.001,
    'gamma': 0.99,
    'buffer_size': 10000,
    'epsilon_min': 0.01,
    'epsilon_max': 1.0,
    'epsilon_decay_period': 10000,
    'epsilon_delay_decay': 0,
    'batch_size': 128,
    'gradient_steps': 1,
    'update_target_strategy': 'replace',
    'update_target_freq': 10,
    'update_target_tau': 0.001,
    'criterion': nn.SmoothL1Loss(),
    'monitoring_nb_trials': 0,
    'max_episodes': 500,
    'model_save_freq': 100,
}

if __name__ == "__main__":
    agent = ProjectAgent(env, config)
    episode_rewards = agent.train()
    agent.save('final_model.pth')
