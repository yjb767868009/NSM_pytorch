import os
import torch
import torch.nn as nn

from model.utils.activation_layer import activation_layer


class Encoder(torch.nn.Module):
    def __init__(self, encoder_dims, encoder_activations, encoder_dropout):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.encoder_activations = encoder_activations
        self.encoder_dropout = encoder_dropout
        self.layer_nums = len(encoder_dims) - 1

        self.layer1 = nn.Sequential(nn.Dropout(encoder_dropout),
                                    nn.Linear(encoder_dims[0], encoder_dims[1]),
                                    activation_layer(encoder_activations[0]))
        self.layer2 = nn.Sequential(nn.Dropout(encoder_dropout),
                                    nn.Linear(encoder_dims[1], encoder_dims[2]),
                                    activation_layer(encoder_activations[1]))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def save_network(self, encoder_index, save_path):
        for i in range(self.layer_nums):
            self.state_dict()['layer%0i.1.weight' % (i + 1)].cpu().detach().numpy().tofile(
                os.path.join(save_path, 'encoder%0i_w%0i.bin' % (encoder_index, i)))
            self.state_dict()['layer%0i.1.bias' % (i + 1)].cpu().detach().numpy().tofile(
                os.path.join(save_path, 'encoder%0i_b%0i.bin' % (encoder_index, i)))
