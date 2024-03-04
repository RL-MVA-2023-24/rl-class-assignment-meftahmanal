import random
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

# Define the neural network architecture
def create_dqn(state_dim, num_actions, nb_neurons=256):
    return nn.Sequential(
        nn.Linear(state_dim, nb_neurons),
        nn.ReLU(),
        nn.Linear(nb_neurons, nb_neurons),
        nn.ReLU(),
        nn.Linear(nb_neurons, nb_neurons),
        nn.ReLU(),
        nn.Linear(nb_neurons, nb_neurons),
        nn.ReLU(),
        nn.Linear(nb_neurons, nb_neurons),
        nn.ReLU(),
        nn.Linear(nb_neurons, num_actions)
    )

# Define the replay buffer class
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.buffer = []
        self.index = 0
        self.device = device
    
    def append(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), zip(*batch)))
    
    def __len__(self):
        return len(self.buffer)

# Define the DQN agent class
class ProjectAgent:
    def __init__(self, config, model):
        self.config = config
        self.device = config['device']
        self.num_actions = config['num_actions']
        self.model = model
        self.target_model = deepcopy(self.model).to(self.device)
        self.memory = ReplayBuffer(config['buffer_size'], self.device)
        self.optimizer = config['optimizer'](self.model.parameters(), lr=config['learning_rate'])
        self.criterion = config['criterion']
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.num_gradient_steps = config['gradient_steps']
        self.update_target_strategy = config['update_target_strategy']
        self.update_target_freq = config['update_target_freq']
        self.update_target_tau = config['update_target_tau']
        self.monitoring_num_trials = config['monitoring_num_trials']

    def predict_action(self, observation, use_random=False):
        return self.select_greedy_action(self.model, observation)

    def store(self, path):
        torch.save(self.model.state_dict(), path)

    def retrieve(self, path):
        self.model.load_state_dict(torch.load(path))

    def evaluate_monte_carlo(self, env, num_trials):
        total_rewards = []
        discounted_rewards = []
        for _ in range(num_trials):
            state, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or truncated):
                action = self.select_greedy_action(self.model, state)
                next_state, reward, done, truncated, _ = env.step(action)
                state = next_state
                total_reward += reward
                discounted_reward += self.gamma ** step * reward
                step += 1
            total_rewards.append(total_reward)
            discounted_rewards.append(discounted_reward)
        return np.mean(discounted_rewards), np.mean(total_rewards)

    def initial_state_value(self, env, num_trials):
        with torch.no_grad():
            values = []
            for _ in range(num_trials):
                state, _ = env.reset()
                values.append(self.model(torch.Tensor(state).unsqueeze(0).to(self.device)).max().item())
        return np.mean(values)

    def update_gradient(self):
        if len(self.memory) > self.batch_size:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            q_next_max = self.target_model(next_states).max(1)[0].detach()
            update = torch.addcmul(rewards, 1 - dones, q_next_max, value=self.gamma)
            q_pred = self.model(states).gather(1, actions.to(torch.long).unsqueeze(1))
            loss = self.criterion(q_pred, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def select_greedy_action(self, network, state):
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()

    def train_agent(self, env, max_episodes):
        episode_returns = []
        mc_avg_total_rewards = []
        mc_avg_discounted_rewards = []
        initial_state_values = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episodes:
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.select_greedy_action(self.model, state)
            next_state, reward, done, truncated, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            for _ in range(self.num_gradient_steps):
                self.update_gradient()
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau * model_state_dict[key] + (1 - tau) * target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            step += 1
            if done or truncated:
                episode += 1
                if self.monitoring_num_trials > 0:
                    mc_discounted_reward, mc_total_reward = self.evaluate_monte_carlo(env, self.monitoring_num_trials)
                    initial_state_value = self.initial_state_value(env, self.monitoring_num_trials)
                    mc_avg_total_rewards.append(mc_total_reward)
                    mc_avg_discounted_rewards.append(mc_discounted_reward)
                    initial_state_values.append(initial_state_value)
                    episode_returns.append(episode_cum_reward)
                    print(f"Episode {episode:2d}, epsilon {epsilon:.2f}, batch size {len(self.memory):4d}, ep return {episode_cum_reward:.2f}, MC tot {mc_total_reward:.2f}, MC disc {mc_discounted_reward:.2f}, V0 {initial_state_value:.2f}")
                else:
                    episode_returns.append(episode_cum_reward)
                    print(f"Episode {episode:2d}, epsilon {epsilon:.2f}, batch size {len(self.memory):4d}, ep return {episode_cum_reward:.2f}")
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_returns, mc_avg_discounted_rewards, mc_avg_total_rewards, initial_state_values

# Configuration parameters
config = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_actions': env.action_space.n,
    'learning_rate': 1e-3,
    'gamma': 0.95,
    'buffer_size': 1000000,
    'epsilon_min': 0.01,
    'epsilon_max': 1.,
    'epsilon_decay_period': 1000,
    'epsilon_delay_decay': 20,
    'batch_size': 512,
    'gradient_steps': 3,
    'update_target_strategy': 'replace',  # or 'ema'
    'update_target_freq': 50,
    'update_target_tau': 0.005,
    'criterion': nn.SmoothL1Loss(),
    'monitoring_num_trials': 0,
    'optimizer': torch.optim.Adam,
}

# Initialize the agent
state_dim = env.observation_space.shape[0]
agent_model = create_dqn(state_dim, config['num_actions'])
agent = ProjectAgent(config, agent_model)

# Train the agent
ep_length, disc_rewards, tot_rewards, V0 = agent.train_agent(env, 200)

# Save the trained model
agent.store("model.pt")

# Evaluate the agent
score_agent = evaluate_HIV(agent=agent, nb_episode=1)
score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=15)
