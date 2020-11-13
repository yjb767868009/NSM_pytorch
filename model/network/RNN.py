import os
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
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
        self.lstm1 = nn.LSTM(rnn_dims[2], rnn_dims[3], batch_first=True)
        self.lstm2 = nn.LSTM(rnn_dims[3], rnn_dims[4], batch_first=True)
        self.fc3 = nn.Sequential(nn.Dropout(rnn_dropout),
                                 nn.Linear(rnn_dims[4], rnn_dims[5]),
                                 activation_layer(rnn_activations[2])
                                 )

    def forward(self, x, x_length):
        # x.size (batch_size,max_data_length,dim)
        x = self.fc1(x)
        x = self.fc2(x)
        x = rnn_utils.pack_padded_sequence(x, x_length, batch_first=True)
        x, (h_1, c_1) = self.lstm1(x)
        x, (h_2, c_2) = self.lstm2(x)
        x, x_length = rnn_utils.pad_packed_sequence(x, batch_first=True)
        x = self.fc3(x)
        return x

    def save_network(self):
        pass
