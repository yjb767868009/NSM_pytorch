import torch
import torch.nn as nn

from model.utils.activation_layer import activation_layer
import os


class Expert(nn.Module):
    def __init__(self, expert_nums, expert_dims, expert_activations, expert_keep_prob):
        super().__init__()
        self.expert_dims = expert_dims
        self.expert_activations = expert_activations
        self.expert_keep_prob = expert_keep_prob
        self.expert_nums = expert_nums
        self.layer_nums = len(expert_dims) - 1

        self.W = []
        self.B = []
        self.D = []
        self.A = []
        for i in range(self.layer_nums):
            w = torch.randn(self.expert_nums, self.expert_dims[i + 1], self.expert_dims[i]).cuda()
            self.W.append(nn.Parameter(w))
            b = torch.zeros(self.expert_nums, self.expert_dims[i + 1], 1).cuda()
            self.B.append(nn.Parameter(b))
            self.D.append(nn.Dropout(p=expert_keep_prob))
            self.A.append(activation_layer(self.expert_activations[i]))

    def forward(self, weight_blend, x):
        for i in range(self.layer_nums):
            x = self.D[i](x)
            x = x.unsqueeze(-1)
            batch_nums = weight_blend.size()[0]
            c = weight_blend.unsqueeze(-1).unsqueeze(-1)
            w = self.W[i].unsqueeze(0)
            w_size = w.size()
            w = w.expand(batch_nums, w_size[1], w_size[2], w_size[3])
            weight = c * w
            weight = weight.sum(dim=1)
            t = torch.bmm(weight, x)

            d = weight_blend.unsqueeze(-1).unsqueeze(-1)
            b = self.B[i].unsqueeze(0)
            b_size = b.size()
            b = b.expand(batch_nums, b_size[1], b_size[2], b_size[3])
            bias = d * b
            bias = bias.sum(dim=1)
            x = torch.add(t, bias)
            x = x.squeeze(-1)

            if self.A[i]:
                if self.A[i] == activation_layer('softmax'):
                    x = self.A[i](x)
                else:
                    x = self.A[i](x)
        return x

    def save_network(self, expert_index, save_path):
        for i in range(self.layer_nums):
            for j in range(self.expert_nums):
                torch.save(self.W[i], os.path.join(save_path, 'wc%0i%0i%0i_w.bin' % (expert_index, i, j)))
                torch.save(self.B[i], os.path.join(save_path, 'wc%0i%0i%0i_b.bin' % (expert_index, i, j)))
