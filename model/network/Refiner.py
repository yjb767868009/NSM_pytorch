import os
import torch
import torch.nn as nn

from model.utils.activation_layer import *


class Refiner(nn.Module):
    """
    Refiner Posture,Trajectory,Goal
    """

    def __init__(self, refiner_dims, refiner_activations, refiner_dropout):
        super().__init__()
        self.layer = nn.Sequential(nn.Dropout(refiner_dropout[0]),
                                   nn.Linear(refiner_dims[0], refiner_dims[1]),
                                   activation_layer(refiner_activations[0]),
                                   nn.LSTM(refiner_dims[1], refiner_dims[2]),
                                   nn.Dropout(refiner_dropout[1]),
                                   nn.Linear(refiner_dims[2], refiner_dims[3]),
                                   activation_layer(refiner_activations[0]))

    def forward(self, x):
        x = self.layer(x)
        return x

    def save_network(self):
        pass
