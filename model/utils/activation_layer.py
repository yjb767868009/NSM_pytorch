import torch.nn as nn

activation_layer_list = {
    'elu': nn.ELU(),
    'softmax': nn.Softmax(),
    'None': None
}


def activation_layer(s):
    return activation_layer_list.get(s)
