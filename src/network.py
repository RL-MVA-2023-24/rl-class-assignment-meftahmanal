import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, nb_neurons=50):
        super(NeuralNetwork, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_size, nb_neurons),
            nn.SELU(),
            nn.Linear(nb_neurons, 2 * nb_neurons),
            nn.SELU(),
            nn.Linear(2 * nb_neurons, nb_neurons),
            nn.SELU(),
            nn.Linear(nb_neurons, output_size)
        )

    def forward(self, x):
        return self.sequence(x)
