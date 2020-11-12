import os
import torch
import torch.nn as nn

from ..utils.activation_layer import activation_layer


class Discriminative(nn.Module):
    def __init__(self, discriminative_dims, discriminative_activations, discriminative_dropout):
        super().__init__()
        self.lstm_hidden_size = discriminative_dims[2]
        self.fc1 = nn.Sequential(nn.Linear(discriminative_dims[0], discriminative_dims[1]),
                                 activation_layer(discriminative_activations[0]))
        self.lstm = nn.LSTM(discriminative_dims[1], discriminative_dims[2], num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc2 = nn.Sequential(nn.Linear(discriminative_dims[2] * 2, discriminative_dims[3]),
                                 activation_layer(discriminative_activations[1]))
        self.fc3 = nn.Sequential(nn.Linear(discriminative_dims[3], discriminative_dims[4]),
                                 activation_layer(discriminative_activations[2]))

    def forward(self, x):
        x = self.fc1(x)
        x, _ = self.lstm(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x[:, -1, 0]
