import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils
import torch.utils.cpp_extension
import torch.utils.data as tordata

from .network import Expert, Encoder

print('CUDA_HOME:', torch.utils.cpp_extension.CUDA_HOME)  # 输出 Pytorch 运行时使用的 cuda
print('torch cuda version:', torch.version.cuda)
print('cuda is available:', torch.cuda.is_available())


class Model(object):
    def __init__(self,
                 model_name, epoch, batch_size, segmentation,
                 train_source, test_source,
                 encoder_nums, encoder_dims, encoder_activations, encoder_keep_prob,
                 expert_components, expert_dims, expert_activations, expert_keep_prob,
                 lr,
                 ):
        self.model_name = model_name
        self.epoch = epoch
        self.batch_size = batch_size
        self.segmentation = segmentation

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

        self.weight_blend_init = torch.Tensor([1]).cuda()

        params_list = []
        for e in self.encoders:
            params_list.append({'params': e.parameters()})
        for e in self.experts:
            params_list.append({'params': e.parameters()})
        self.lr = lr
        self.optimizer = optim.AdamW(params_list,
                                     lr=self.lr)

        self.loss_function = nn.SmoothL1Loss()

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
            for index, (x, y) in enumerate(train_loader):
                batch_nums = x.size()[0]
                weight_blend_first = self.weight_blend_init.unsqueeze(0).expand(batch_nums, 1)
                self.optimizer.zero_grad()
                status_outputs = []
                for i, encoder in enumerate(self.encoders):
                    status_output = encoder(x[:, self.segmentation[i]:self.segmentation[i + 1]])
                    status_outputs.append(status_output)
                status = torch.cat(tuple(status_outputs), 1)

                expert_first = self.experts[0]
                weight_blend = expert_first(weight_blend_first, x[:, self.segmentation[-2]:self.segmentation[-1]])
                #print('expert_first weight_blend', weight_blend)

                expert_last = self.experts[-1]
                output = expert_last(weight_blend, status)
                #print('output', output)
                y = y.cuda()
                loss = self.loss_function(output, y)
                loss_list.append(loss.item())

                loss.backward()
                self.optimizer.step()
            # print(loss_list)
            avg_loss = np.asarray(loss_list).mean()
            train_loss.append(avg_loss)
            print('Epoch {}:', format(e + 1), 'Training Loss =', '{:.9f}'.format(avg_loss))
