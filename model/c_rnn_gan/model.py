import logging
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

from ..network import *

# Check GPU available
print('CUDA_HOME:', torch.utils.cpp_extension.CUDA_HOME)
print('torch cuda version:', torch.version.cuda)
print('cuda is available:', torch.cuda.is_available())


def build_network(name, dims, activations, dropout):
    network = eval(name)(dims, activations, dropout)
    if torch.cuda.is_available():
        network.cuda()
    network = nn.DataParallel(network)
    return network


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
                 # For Refiner network information
                 refiner_dims, refiner_activations, refiner_dropout,
                 # For Discriminative network information
                 discriminative_dims, discriminative_activations, discriminative_dropout,
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
            self.weight_blend_init = self.weight_blend_init.cuda()

        # build Refiner network
        self.refiner = build_network('Refiner', refiner_dims, refiner_activations, refiner_dropout)

        # build Discriminative network
        self.discriminative = build_network('Discriminative',
                                            discriminative_dims, discriminative_activations,
                                            discriminative_dropout)

        # build optimizer
        params_list = []
        for e in self.encoders:
            params_list.append({'params': e.parameters()})
        for e in self.experts:
            params_list.append({'params': e.parameters()})
        self.lr = lr
        self.optimizer = optim.AdamW(params_list,
                                     lr=self.lr)

        self.refiner_optimizer = optim.RMSprop(self.refiner.parameters())
        self.discriminative_optimizer = optim.RMSprop(self.discriminative.parameters())

        # build loss function
        self.loss_function = nn.MSELoss(reduction='mean')
        # todo refiner and discriminative loss
        self.refiner_loss_function = nn.BCELoss()
        self.discriminative_loss_function = nn.BCELoss()

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))

    def load(self):
        print('Loading parm...')
        for i in range(self.encoder_nums):
            self.encoders[i].load_state_dict(torch.load(os.path.join(self.load_path, 'encoder%0i.pth' % i)))
        for i in range(self.expert_nums):
            self.experts[i].load_state_dict(torch.load(os.path.join(self.load_path, 'expert%0i.pth' % i)))
        self.optimizer.load_state_dict(torch.load(os.path.join(self.load_path, 'optimizer.ptm')))
        print('Loading param complete')

    def train(self):
        print("Training START")
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
            for x, y in tqdm(train_loader, ncols=100):
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
        print("Training COMPLETE")

    def train_gan(self):
        self.load()
        print("Training GAN")

        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
        )
        for encoder in self.encoders:
            encoder.eval()
        for expert in self.experts:
            expert.eval()

        train_refiner_loss = []
        train_discriminative_loss = []
        for e in range(self.epoch):
            refiner_loss_list = []
            discriminative_loss_list = []
            if e % 50 == 0:
                self.lr = self.lr / 10
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            for x, y in tqdm(train_loader, ncols=100):
                batch_nums = x.size()[0]

                # Generate NSM output
                weight_blend_first = self.weight_blend_init.unsqueeze(0).expand(batch_nums, 1)
                status_outputs = []
                for i, encoder in enumerate(self.encoders):
                    status_output = encoder(x[:, self.segmentation[i]:self.segmentation[i + 1]])
                    status_outputs.append(status_output)
                status = torch.cat(tuple(status_outputs), 1)
                expert_first = self.experts[0]
                weight_blend = expert_first(weight_blend_first, x[:, self.segmentation[-2]:self.segmentation[-1]])
                expert_last = self.experts[-1]
                output = expert_last(weight_blend, status)

                # Train Discriminative Network
                self.refiner_optimizer.zero_grad()
                self.discriminative_optimizer.zero_grad()

                # Generate real and fake data label
                real_label = torch.autograd.Variable(torch.ones(batch_nums))
                fake_label = torch.autograd.Variable(torch.zeros(batch_nums))
                if torch.cuda.is_available():
                    real_label = real_label.cuda()
                    fake_label = fake_label.cuda()

                # Real data's loss
                real_out = self.discriminative(y)
                discriminative_real_loss = self.discriminative_loss_function(real_out, real_label)
                discriminative_loss_list.append(discriminative_real_loss.item())

                # Fake data's loss
                fake_data = self.refiner(output)
                fake_out = self.discriminative(fake_data)
                discriminative_fake_loss = self.discriminative_loss_function(fake_out, fake_label)

                # loss backward and renew optimizer
                discriminative_loss = discriminative_real_loss + discriminative_fake_loss
                discriminative_loss.backward()
                self.discriminative_optimizer.step()

                # Train Refiner Network
                fake_data = self.refiner(output)
                fake_out = self.discriminative(fake_data)
                refiner_loss = self.refiner_loss_function(fake_out, real_label)
                refiner_loss_list.append(refiner_loss.item())
                refiner_loss.backward()
                self.refiner_optimizer.step()

            avg_refiner_loss = np.asarray(refiner_loss_list).mean()
            train_refiner_loss.append(avg_refiner_loss)
            avg_discriminative_loss = np.asarray(discriminative_loss_list).mean()
            train_discriminative_loss.append(avg_discriminative_loss)

            print('Time {} '.format(datetime.datetime.now()),
                  'Epoch {} : '.format(e + 1),
                  'Refiner Loss = {:.9f} '.format(avg_refiner_loss),
                  'Discriminative Loss = {:.9f} '.format(avg_discriminative_loss),
                  'lr = {} '.format(self.lr),
                  )
            logging.info('Epoch {} : '.format(e + 1) +
                         'Refiner Loss = {:.9f} '.format(avg_refiner_loss) +
                         'Discriminative Loss = {:.9f} '.format(avg_discriminative_loss) +
                         'lr = {} '.format(self.lr))
        print('Training GAN COMPLETE')

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
        for x, y in tqdm(train_loader, ncols=100):
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
