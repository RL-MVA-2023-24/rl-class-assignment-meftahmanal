import numpy as np
from collections import defaultdict

# Import the environment and necessary wrappers
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

# Set up the environment with domain randomization disabled
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)

class ProjectAgent:
    def __init__(self, n_actions=4, n_states=6, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def act(self, state, use_random=False):
        if np.random.uniform(0, 1) < self.epsilon or use_random:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        if not done:
            future_reward = np.max(self.q_table[next_state])
        else:
            future_reward = 0
        td_target = reward + self.gamma * future_reward
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path, allow_pickle=True).item()

def train(agent, env, episodes=1000):
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(str(obs))
            next_obs, reward, done, _, _ = env.step(action)
            agent.learn(str(obs), action, reward, str(next_obs), done)
            obs = next_obs
            total_reward += reward
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    return agent

if __name__ == "__main__":
    # Initialize the agent
    agent = ProjectAgent()

    # Train the agent
    trained_agent = train(agent, env)

    # Save the trained agent
    trained_agent.save("trained_agent.npy")
    print("Agent trained and saved.")