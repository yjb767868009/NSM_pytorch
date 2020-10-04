import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(message)s',
                    filename='log.txt')

import os
import numpy as np
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.cpp_extension
import torch.utils.data as tordata

from .network import Expert, Encoder

# Check GPU available
print('CUDA_HOME:', torch.utils.cpp_extension.CUDA_HOME)
print('torch cuda version:', torch.version.cuda)
print('cuda is available:', torch.cuda.is_available())


class Model(object):
    def __init__(self,
                 # For Model base information
                 model_name, epoch, batch_size, segmentation, save_path, load_path,
                 # For Date information
                 train_source, test_source,
                 # For encoder network information
                 encoder_nums, encoder_dims, encoder_activations, encoder_dropout,
                 # For expert network information
                 expert_components, expert_dims, expert_activations, expert_dropout,
                 # optim param
                 lr,
                 ):
        self.model_name = model_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.segmentation = segmentation
        self.save_path = save_path
        self.load_path = load_path

        self.train_source = train_source
        self.test_source = test_source

        # build encoder network
        self.encoder_nums = encoder_nums
        self.encoders = []
        for i in range(encoder_nums):
            encoder = Encoder(encoder_dims[i], encoder_activations[i], encoder_dropout)
            if torch.cuda.is_available():
                encoder.cuda()
            encoder = nn.DataParallel(encoder)
            self.encoders.append(encoder)

        # build expert network
        self.expert_nums = len(expert_components)
        self.experts = []
        for i in range(self.expert_nums):
            expert = Expert(expert_components[i], expert_dims[i], expert_activations[i], expert_dropout)
            if torch.cuda.is_available():
                expert.cuda()
            expert = nn.DataParallel(expert)
            self.experts.append(expert)

        # weight blend init
        self.weight_blend_init = torch.Tensor([1])
        if torch.cuda.is_available():
            self.weight_blend_init=self.weight_blend_init.cuda()

        # build optimizer
        params_list = []
        for e in self.encoders:
            params_list.append({'params': e.parameters()})
        for e in self.experts:
            params_list.append({'params': e.parameters()})
        self.lr = lr
        self.optimizer = optim.AdamW(params_list,
                                     lr=self.lr)

        # build loss function
        self.loss_function = nn.SmoothL1Loss()

    def load(self):
        print('Loading parm...')
        for i in range(self.encoder_nums):
            self.encoders[i].load_state_dict(torch.load(os.path.join(self.load_path, 'encoder%0i.pth' % i)))
        for i in range(self.expert_nums):
            self.experts[i].load_state_dict(torch.load(os.path.join(self.load_path, 'expert%0i.pth' % i)))
        self.optimizer.load_state_dict(torch.load(os.path.join(self.load_path, 'optimizer.ptm')))
        print('Loading param complete')

    def train(self):
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
        )
        for encoder in self.encoders:
            encoder.train()
        for expert in self.experts:
            expert.train()

        train_loss = []
        for e in range(self.epoch):
            loss_list = []
            if e % 50 == 0:
                self.lr = self.lr / 10
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            for x, y in tqdm(train_loader,ncols=100):
                batch_nums = x.size()[0]
                weight_blend_first = self.weight_blend_init.unsqueeze(0).expand(batch_nums, 1)
                self.optimizer.zero_grad()
                status_outputs = []
                for i, encoder in enumerate(self.encoders):
                    status_output = encoder(x[:, self.segmentation[i]:self.segmentation[i + 1]])
                    status_outputs.append(status_output)
                status = torch.cat(tuple(status_outputs), 1)

                # Gating Network
                expert_first = self.experts[0]
                weight_blend = expert_first(weight_blend_first, x[:, self.segmentation[-2]:self.segmentation[-1]])

                # Motion Network
                expert_last = self.experts[-1]
                output = expert_last(weight_blend, status)

                # loss
                if torch.cuda.is_available():
                    y = y.cuda()
                loss = self.loss_function(output, y)
                loss_list.append(loss.item())

                loss.backward()
                self.optimizer.step()
            if e % 10 == 0:
                # save param for unity
                for i in range(self.encoder_nums):
                    self.encoders[i].module.save_network(i, self.save_path)
                for i in range(self.expert_nums):
                    self.experts[i].module.save_network(i, self.save_path)
                # save model for load weights
                for i in range(self.encoder_nums):
                    torch.save(self.encoders[i].state_dict(), os.path.join(self.save_path, 'encoder%0i.pth' % i))
                for i in range(self.expert_nums):
                    torch.save(self.experts[i].state_dict(), os.path.join(self.save_path, 'expert%0i.pth' % i))
                torch.save(self.optimizer.state_dict(), os.path.join(self.save_path, 'optimizer.ptm'))

            avg_loss = np.asarray(loss_list).mean()
            train_loss.append(avg_loss)
            print('Time {} '.format(datetime.datetime.now()),
                  'Epoch {} : '.format(e + 1),
                  'Training Loss = {:.9f} '.format(avg_loss),
                  'lr = {} '.format(self.lr),
                  )
            logging.info('Epoch {} : '.format(e + 1) +
                         'Training Loss = {:.9f} '.format(avg_loss) +
                         'lr = {} '.format(self.lr))
        torch.save(train_loss, os.path.join(self.save_path, 'trainloss.bin'))
        print('Learning Finished')

    def test(self):
        self.load()
        train_loader = tordata.DataLoader(
            dataset=self.test_source,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
        )
        for encoder in self.encoders:
            encoder.eval()
        for expert in self.experts:
            expert.eval()

        test_loss = []
        for x, y in tqdm(train_loader,ncols=100):
            batch_nums = x.size()[0]
            weight_blend_first = self.weight_blend_init.unsqueeze(0).expand(batch_nums, 1)
            status_outputs = []
            for i, encoder in enumerate(self.encoders):
                status_output = encoder(x[:, self.segmentation[i]:self.segmentation[i + 1]])
                status_outputs.append(status_output)
            status = torch.cat(tuple(status_outputs), 1)

            # Gating Network
            expert_first = self.experts[0]
            weight_blend = expert_first(weight_blend_first, x[:, self.segmentation[-2]:self.segmentation[-1]])

            # Motion Network
            expert_last = self.experts[-1]
            output = expert_last(weight_blend, status)

            # loss
            if torch.cuda.is_available():
                y = y.cuda()
            loss = self.loss_function(output, y)
            test_loss.append(loss.item())

        avg_loss = np.asarray(test_loss).mean()
        print('Testing Loss = {:.9f} '.format(avg_loss))
        print('Testing Finished')
