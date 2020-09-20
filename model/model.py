import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata

import numpy
from .network import NSM, Expert, Encoder


class Model(object):
    def __init__(self,
                 model_name, epoch, batch_size,
                 train_source, test_source,
                 encoder_nums, encoder_dims, encoder_activations, encoder_keep_prob,
                 expert_components, expert_dims, expert_activations, expert_keep_prob,
                 lr,
                 ):
        self.model_name = model_name
        self.epoch = epoch
        self.batch_size = batch_size

        self.train_source = train_source
        self.test_source = test_source

        self.encoder_nums = encoder_nums
        self.encoders = []
        for i in range(encoder_nums):
            encoder = Encoder(encoder_dims[i], encoder_activations[i], encoder_keep_prob)
            encoder = nn.DataParallel(encoder)
            encoder.cuda()
            self.encoders.append(encoder)

        self.expert_nums = len(expert_components)
        self.experts = []
        for i in range(self.expert_nums):
            expert = Expert(expert_components[i], expert_dims[i], expert_activations[i], expert_keep_prob)
            expert = nn.DataParallel(expert)
            expert.cuda()
            self.experts.append(expert)

        self.lr = lr
        self.optimizer = optim.AdamW(
                                     lr=self.lr)

    def train(self):
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
        )
        for e in range(self.epoch):
            for index, (x, y) in enumerate(train_loader):
                for i in range(self.encoder_nums):
                    pass
