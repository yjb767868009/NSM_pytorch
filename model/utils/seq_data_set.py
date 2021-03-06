import torch
import torch.utils.data as tordata
import numpy as np
import os
from tqdm import tqdm


class DataSet(tordata.Dataset):

    def __init__(self, input_dir, label_dir, cache, ):
        self.input_data_dir = input_dir
        assert os.path.exists(self.input_data_dir), 'No Input dir'
        self.label_data_dir = label_dir
        assert os.path.exists(self.label_data_dir), 'No label dir'

        self.data_size = len(os.listdir(self.input_data_dir))
        assert self.data_size == len(os.listdir(self.label_data_dir)), 'input size!= label size'

        self.input_data = [None for _ in range(self.data_size + 1)]
        self.label_data = [None for _ in range(self.data_size + 1)]

        self.cache = cache

    def load_data(self, index):
        return self.__getitem__(index)

    def load_all_data(self):
        for i in tqdm(range(self.data_size)):
            self.load_data(i)

    def __len__(self):
        return self.data_size

    def __loader__(self, path):
        return torch.FloatTensor(np.float32(np.loadtxt(path)))

    def __getitem__(self, item):
        if not self.cache:
            input_data = self.__loader__(os.path.join(self.input_data_dir, str(item) + '.txt'))
            label_data = self.__loader__(os.path.join(self.label_data_dir, str(item) + '.txt'))
        elif self.input_data[item] is None or self.label_data[item] is None:
            input_data = self.__loader__(os.path.join(self.input_data_dir, str(item) + '.txt'))
            label_data = self.__loader__(os.path.join(self.label_data_dir, str(item) + '.txt'))
            self.input_data[item] = input_data
            self.label_data[item] = label_data
        else:
            input_data = self.input_data[item]
            label_data = self.label_data[item]
        return input_data, label_data
