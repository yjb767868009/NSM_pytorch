import os
import torch
import torch.nn as nn

from ..utils.activation_layer import activation_layer


class RNN(nn.Module):
    def __init__(self, rnn_dims, rnn_activations, rnn_dropout):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Dropout(rnn_dropout),
                                 nn.Linear(rnn_dims[0], rnn_dims[1]),
                                 activation_layer(rnn_activations[0]), )
        self.fc2 = nn.Sequential(nn.Dropout(rnn_dropout),
                                 nn.Linear(rnn_dims[1], rnn_dims[2]),
                                 activation_layer(rnn_activations[1]), )
        self.lstm1 = nn.LSTM(rnn_dims[2], rnn_dims[3])
        self.lstm2 = nn.LSTM(rnn_dims[3], rnn_dims[4])
        self.fc3 = nn.Sequential(nn.Dropout(rnn_dropout),
                                 nn.Linear(rnn_dims[4], rnn_dims[5]),
                                 activation_layer(rnn_activations[2])
                                 )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x, (h_1, c_1) = self.lstm1(x)
        x, (h_2, c_2) = self.lstm2(x)
        x = self.fc3(x)
        return x

    def save_network(self):
        pass
