import torch.nn as nn

activation_layer_list = {
    'relu': nn.ReLU(),
    'elu': nn.ELU(),
    'softmax': nn.Softmax(dim=1),
    'None': None
}


def activation_layer(s):
    return activation_layer_list.get(s)
