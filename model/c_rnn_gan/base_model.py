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
import torch.nn.utils.rnn as rnn_utils
from ..network import *
from ..utils import build_network


class BaseModel(object):
    def __init__(self,
                 # For Model base information
                 model_name, epoch, batch_size, segmentation, save_path, load_path, lr, save_output,
                 # For Date information
                 train_source, test_source,
                 # For encoder network information
                 encoder_nums, encoder_dims, encoder_activations, encoder_dropout,
                 # For RNN network information
                 rnn_dims, rnn_activations, rnn_dropout,
                 ):
        self.model_name = model_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.segmentation = segmentation
        self.save_path = save_path
        self.load_path = load_path
        self.save_output = save_output

        self.train_source = train_source
        self.test_source = test_source

        # build encoder network
        self.encoder_nums = encoder_nums
        self.encoders = []
        for i in range(encoder_nums):
            encoder = build_network('Encoder', encoder_dims[i], encoder_activations[i], encoder_dropout)
            self.encoders.append(encoder)

        # build RNN network
        self.rnn = build_network('RNN', rnn_dims, rnn_activations, rnn_dropout)

        # build optimizer
        params_list = []
        for e in self.encoders:
            params_list.append({'params': e.parameters()})
        self.lr = lr
        self.encoder_optimizer = optim.AdamW(params_list,
                                             lr=self.lr)
        self.rnn_optimizer = optim.Adam(self.rnn.parameters())
        # build loss function
        self.loss_function = nn.MSELoss(reduction='mean')

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))

    def collate_fn(self, data):
        batch_size = len(data)
        input_data = [data[i][0] for i in range(batch_size)]
        output_data = [data[i][1] for i in range(batch_size)]
        input_data.sort(key=lambda x: len(x), reverse=True)
        output_data.sort(key=lambda x: len(x), reverse=True)
        data_length = [len(sq) for sq in input_data]
        input_data = rnn_utils.pad_sequence(input_data, batch_first=True, padding_value=0)
        output_data = rnn_utils.pad_sequence(output_data, batch_first=True, padding_value=0)
        return [input_data, output_data], data_length

    def load_param(self):
        print('Loading parm...')
        # Load Model
        for i in range(self.encoder_nums):
            self.encoders[i].load_state_dict(torch.load(os.path.join(self.load_path, 'encoder%0i.pth' % i)))
        self.rnn.load_state_dict(torch.load(os.path.join(self.load_path, 'rnn.pth')))
        # Load optimizer
        self.encoder_optimizer.load_state_dict(torch.load(os.path.join(self.load_path, 'encoder_optimizer.ptm')))
        self.rnn_optimizer.load_state_dict(torch.load(os.path.join(self.load_path, 'rnn_optimizer.ptm')))
        print('Loading param complete')
        for encoder in self.encoders:
            encoder.eval()
        self.rnn.eval()

    def forward(self, x, x_length=None):
        # Encoder Network
        status_outputs = []
        for i, encoder in enumerate(self.encoders):
            status_output = encoder(x[:, :, self.segmentation[i]:self.segmentation[i + 1]])
            status_outputs.append(status_output)
        status = torch.cat(tuple(status_outputs), 2)

        # RNN Network
        output = self.rnn(status, x_length)
        return output

    def train(self):
        print("Training START")
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        for encoder in self.encoders:
            encoder.train()
        self.rnn.train()

        train_loss = []
        for e in range(self.epoch):
            loss_list = []
            if e % 50 == 0:
                self.lr = self.lr / 10
                for param_group in self.encoder_optimizer.param_groups:
                    param_group['lr'] = self.lr
            for data, data_length in tqdm(train_loader, ncols=100):
                x = data[0]
                y = data[1]
                batch_nums = x.size(0)
                self.encoder_optimizer.zero_grad()
                self.rnn_optimizer.zero_grad()

                output = self.forward(x, data_length)

                # loss
                if torch.cuda.is_available():
                    y = y.cuda()
                loss = self.loss_function(output, y)
                loss_list.append(loss.item())

                loss.backward()
                self.encoder_optimizer.step()
                self.rnn_optimizer.step()
            if e % 10 == 0:
                for i in range(self.encoder_nums):
                    torch.save(self.encoders[i].state_dict(), os.path.join(self.save_path, 'encoder%0i.pth' % i))
                torch.save(self.rnn.state_dict(), os.path.join(self.save_path, 'rnn.pth'))
                torch.save(self.encoder_optimizer.state_dict(), os.path.join(self.save_path, 'encoder_optimizer.ptm'))
                torch.save(self.rnn_optimizer.state_dict(), os.path.join(self.save_path, 'rnn_optimizer.ptm'))

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
        print("Training COMPLETE")

        if self.save_output:
            print("Saving output")
            self.test(self.save_output)
            print("Saving COMPLETE")

    def test(self, save_path=None):
        if save_path:
            save_index = 0
            if not os.path.exists(save_path):
                os.mkdir(save_path)

        self.load_param()

        train_loader = tordata.DataLoader(
            dataset=self.test_source,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=self.collate_fn,
        )

        test_loss = []
        for data, data_length in tqdm(train_loader, ncols=100):
            x = data[0]
            y = data[1]
            batch_nums = x.size(0)

            # Generate nsm output
            output = self.forward(x, data_length)

            if save_path:
                for i in range(batch_nums):
                    np.savetxt(os.path.join(save_path, str(save_index) + ".txt"),
                               output[0, :, :].cpu().detach().numpy())
                    save_index += 1

            # loss
            if torch.cuda.is_available():
                y = y.cuda()
            loss = self.loss_function(output, y)
            test_loss.append(loss.item())

        avg_loss = np.asarray(test_loss).mean()
        print('Testing Loss = {:.9f} '.format(avg_loss))
        print('Testing Finished')
