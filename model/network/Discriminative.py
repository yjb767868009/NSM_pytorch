import os
import torch
import torch.nn as nn

from model.utils.activation_layer import activation_layer


class Discriminative(nn.Module):
    def __init__(self, discriminative_dims, discriminative_activations, discriminative_dropout):
        super().__init__()
        self.lstm_hidden_size = discriminative_dims[2]
        self.layer1 = nn.Sequential(nn.Linear(discriminative_dims[0], discriminative_dims[1]),
                                    activation_layer(discriminative_activations[0]))
        self.lstm = nn.LSTM(discriminative_dims[1], discriminative_dims[2], num_layers=1,
                            batch_first=True, bidirectional=True)
        self.layer2 = nn.Sequential(nn.Linear(discriminative_dims[2], discriminative_dims[3]),
                                    activation_layer(discriminative_activations[1]))

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.lstm_hidden_size)
        c0 = torch.zeros(2, x.size(0), self.lstm_hidden_size)
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        x = self.layer1(x)
        x, _ = self.lstm(x, (h0, c0))
        x = self.layer2(x[:, -1, :])
        return x
