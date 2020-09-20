import torch
import torch.nn as nn

from model.utils.activation_layer import activation_layer


class Expert(nn.Module):
    def __init__(self, expert_weights: torch.Tensor, expert_dims, expert_activations, expert_keep_prob):
        super().__init__()
        self.expert_weights = expert_weights
        self.expert_dims = expert_dims
        self.expert_activations = expert_activations
        self.expert_keep_prob = expert_keep_prob
        self.expert_nums = len(expert_weights)
        self.layer_nums = len(expert_dims) - 1

        self.W = []
        self.B = []
        self.D = []
        self.A = []
        for i in range(self.layer_nums):
            a = torch.Tensor(self.expert_nums, expert_dims[i + 1], expert_dims[i])
            r = self.expert_weights.unsqueeze(-1).unsqueeze(-1)
            weight = r * a
            weight = weight.sum(dim=0)
            self.W.append(nn.Parameter(weight))
            bias = torch.Tensor(expert_dims[1])
            self.B.append(nn.Parameter(bias))
            self.D.append(nn.Dropout(p=expert_keep_prob))
            self.A.append(activation_layer(self.expert_activations[i]))

    def forward(self, x):
        for i in range(self.layer_nums):
            x = self.D[i](x)
            x = torch.add(torch.mm(x, self.W[i].t()), self.B[i])
            if self.A[i]:
                x = self.A[i](x)
        return x
