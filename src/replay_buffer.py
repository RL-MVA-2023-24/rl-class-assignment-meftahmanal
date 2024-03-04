import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.buffer = []
        self.device = device

    def append(self, state, action, reward, next_state, done):
        experience = (torch.tensor([state], dtype=torch.float, device=self.device),
                      torch.tensor([action], dtype=torch.long, device=self.device),
                      torch.tensor([reward], dtype=torch.float, device=self.device),
                      torch.tensor([next_state], dtype=torch.float, device=self.device),
                      torch.tensor([done], dtype=torch.float, device=self.device))
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
