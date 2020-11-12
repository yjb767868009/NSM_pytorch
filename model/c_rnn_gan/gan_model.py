import logging
import os
import numpy as np
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data as tordata

from ..network import *
from ..utils import build_network


class GANModel(object):
    def __init__(self,
                 # For Model base information
                 model_name, epoch, batch_size, save_path, load_path, lr,
                 # For Date information
                 train_source, test_source,
                 # For Refiner network information
                 refiner_dims, refiner_activations, refiner_dropout,
                 # For Discriminative network information
                 discriminative_dims, discriminative_activations, discriminative_dropout,
                 ):
        self.model_name = model_name

        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr

        self.save_path = save_path
        self.load_path = load_path

        self.train_source = train_source
        self.test_source = test_source

        # build Refiner network
        self.refiner = build_network('Refiner', refiner_dims, refiner_activations, refiner_dropout)

        # build Discriminative network
        self.discriminative = build_network('Discriminative', discriminative_dims, discriminative_activations,
                                            discriminative_dropout)

        # todo update refiner and discriminative loss
        self.refiner_loss_function = nn.BCELoss()
        self.discriminative_loss_function = nn.BCELoss()

        self.refiner_optimizer = optim.RMSprop(self.refiner.parameters())
        self.discriminative_optimizer = optim.RMSprop(self.discriminative.parameters())

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))

    def train(self):
        print("Training GAN")

        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
        )

        train_refiner_loss = []
        train_discriminative_loss = []
        for e in range(self.epoch):
            refiner_loss_list = []
            discriminative_loss_list = []
            if e % 50 == 0:
                self.lr = self.lr / 10
                for param_group in self.refiner_optimizer.param_groups:
                    param_group['lr'] = self.lr
                for param_group in self.discriminative_optimizer.param_groups:
                    param_group['lr'] = self.lr
            for x, y in tqdm(train_loader, ncols=100):
                batch_nums = x.size(0)

                # Train Discriminative Network
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
                fake_data = self.refiner(x)
                fake_out = self.discriminative(fake_data)
                discriminative_fake_loss = self.discriminative_loss_function(fake_out, fake_label)

                # discriminative loss backward and renew optimizer
                discriminative_loss = discriminative_real_loss + discriminative_fake_loss
                self.discriminative_optimizer.zero_grad()
                discriminative_loss.backward(retain_graph=True)
                self.discriminative_optimizer.step()

                # Train Refiner Network
                fake_data = self.refiner(x)
                fake_out = self.discriminative(fake_data)
                refiner_loss = self.refiner_loss_function(fake_out, real_label)
                refiner_loss_list.append(refiner_loss.item())

                # refiner loss backward and renew optimizer
                self.refiner_optimizer.zero_grad()
                refiner_loss.backward()
                self.refiner_optimizer.step()

            if e % 10 == 0:
                # save param for unity
                self.refiner.module.save_network(self.save_path)

                # save model for load weights
                torch.save(self.refiner.state_dict(), os.path.join(self.save_path, 'refiner.pth'))
                torch.save(self.discriminative.state_dict(), os.path.join(self.save_path, 'discriminative.pth'))
                torch.save(self.refiner_optimizer.state_dict(), os.path.join(self.save_path, 'refiner_optimizer.ptm'))
                torch.save(self.discriminative_optimizer.state_dict(),
                           os.path.join(self.save_path, 'discriminative_optimizer.ptm'))

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
