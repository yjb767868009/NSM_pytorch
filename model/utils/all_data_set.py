import torch.utils.data as tordata
import numpy as np
import os


class DataSet(tordata.Dataset):
    def __init__(self,
                 input_data, input_dim, input_mean, input_std,
                 output_data, output_dim, output_mean, output_std,
                 data_size, cache, ):
        self.input_data = input_data
        self.input_dim = input_dim
        self.input_mean = input_mean
        self.input_std = input_std

        self.output_data = output_data
        self.output_dim = output_dim
        self.output_mean = output_mean
        self.output_std = output_std

        self.data_size = data_size
        self.cache = cache

    def load_data(self, index):
        return self.__getitem__(index)

    def __getitem__(self, item):
        return self.input_data[item], self.output_data[item]

    def __len__(self):
        return len(self.input_data)


"""
    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def __getitem__(self, item):
        if not self.cache:
            pass
        elif self.data[item] is None:
            data = []
        else:
            data = self.data[item]
"""


def __len__(self):
    return self
