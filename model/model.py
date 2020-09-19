import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata

import numpy
from .network import NSM
from .network.Encoder import Encoder
from .network.Expert import Expert


class Model(object):
    def __init__(self,
                 train_source, test_source,
                 encoder_nums, encoder_dims, encoder_activations, encoder_keep_prob,
                 ):
        self.train_source = train_source
        self.test_source = test_source

        self.encoders = []
        for i in range(encoder_nums):
            encoder = Encoder(encoder_dims[i], encoder_activations[i], encoder_keep_prob)
            encoder = nn.DataParallel(encoder)
            encoder.cuda()
            self.encoders.append(encoder)

    def train(self):
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            num_workers=4,
        )

        pass
