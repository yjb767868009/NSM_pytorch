import torch
import torch.nn as nn

activation_layer_list = {
    'elu': nn.ELU(),
}


def activation_layer(s):
    return activation_layer_list.get(s)


class Encoder(torch.nn.Module):
    def __init__(self, encoder_dims, encoder_activations, encoder_keep_prob):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.encoder_activations = encoder_activations
        self.encoder_keep_prob = encoder_keep_prob

        self.layer1 = nn.Sequential(nn.Dropout(encoder_keep_prob),
                                    nn.Linear(encoder_dims[0], encoder_dims[1]),
                                    activation_layer(encoder_activations[0]))
        self.layer2 = nn.Sequential(nn.Dropout(encoder_keep_prob),
                                    nn.Linear(encoder_dims[1], encoder_dims[2]),
                                    activation_layer(encoder_activations[1]))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
