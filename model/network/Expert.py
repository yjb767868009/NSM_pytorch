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
            w = torch.Tensor(self.expert_nums, self.expert_dims[i + 1], self.expert_dims[i])
            self.W.append(nn.Parameter(w))
            bias = torch.Tensor(expert_dims[1])
            self.B.append(nn.Parameter(bias))
            self.D.append(nn.Dropout(p=expert_keep_prob))
            self.A.append(activation_layer(self.expert_activations[i]))

    def forward(self, weight_blend, x):
        for i in range(self.layer_nums):
            x = self.D[i](x)
            c = weight_blend.unsqueeze(-1).unsqueeze(-1)
            weight = c * self.W[i]
            weight = weight.sum(dim=0)
            x = torch.add(torch.mm(x, weight.t()), self.B[i])
            if self.A[i]:
                x = self.A[i](x)
        return x
