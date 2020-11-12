import torch
import torch.nn as nn

from ..network import *


def build_network(name, dims, activations, dropout):
    network = eval(name)(dims, activations, dropout)
    if torch.cuda.is_available():
        network.cuda()
    network = nn.DataParallel(network)
    return network
