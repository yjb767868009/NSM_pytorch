import torch
import torch.nn as nn

from model.utils.activation_layer import activation_layer


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
            b = torch.zeros(self.expert_nums, self.expert_dims[i + 1]).cuda()
            self.B.append(nn.Parameter(b))
            self.D.append(nn.Dropout(p=expert_keep_prob))
            self.A.append(activation_layer(self.expert_activations[i]))

    def forward(self, weight_blend, x):
        for i in range(self.layer_nums):
            x = self.D[i](x)
            x = x.unsqueeze(-1)
            c = weight_blend.unsqueeze(-1).unsqueeze(-1)
            weight = c * self.W[i]
            weight = weight.sum(dim=1)
            t = torch.bmm(weight, x)
            t = t.squeeze(-1)
            bais = weight_blend.unsqueeze(-1) * self.B[i]
            bais = bais.sum(dim=1)
            x = torch.add(t, bais)
            if self.A[i]:
                if self.A[i] == activation_layer('softmax'):
                    x = self.A[i](x)
                else:
                    x = self.A[i](x)
        return x
