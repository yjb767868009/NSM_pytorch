import torch.utils.data as tordata
import numpy as np
import os


class DataSet(tordata.Dataset):

    def __init__(self, data_dir, cache, ):
        self.data_dir = data_dir
        self.input_data_dir = os.path.join(data_dir, 'Input')
        assert not os.path.exists(self.input_data_dir), 'No Input dir'
        self.output_data_dir = os.path.join(data_dir, 'Output')
        assert not os.path.exists(self.output_data_dir), 'No Output dir'

        self.data_size = len(os.listdir(self.input_data_dir))
        assert self.data_size == len(os.listdir(self.output_data_dir)), 'input size!= output size'

        self.input_data = [None for _ in range(self.data_size)]
        self.output_data = [None for _ in range(self.data_size)]

        self.input_mean, self.input_std = self.get_normalize('InputNorm.txt')
        self.output_mean, self.output_std = self.get_normalize('OutputNorm.txt')

        self.cache = cache

    def get_normalize(self, file_name):
        normalize_data = self.__loader__(os.path.join(self.data_dir, file_name))
        mean = normalize_data[0]
        std = normalize_data[1]
        for i in range(std.size):
            if std[i] == 0:
                std[i] = 1
        return mean, std

    def load_data(self, index):
        return self.__getitem__(index)

    def load_all_data(self):
        for i in range(self.data_size):
            self.load_data(i)

    def __len__(self):
        return self.data_size

    def __loader__(self, path):
        return np.float32(np.loadtxt(path))

    def __getitem__(self, item):
        if not self.cache:
            input_data = self.__loader__(os.path.join(self.input_data_dir, str(item) + '.txt'))
            input_data = (input_data - self.input_mean) / self.input_std
            output_data = self.__loader__(os.path.join(self.output_data_dir, str(item) + '.txt'))
            output_data = (output_data - self.output_mean) / self.output_std
        elif self.input_data[item] is None or self.output_data[item] is None:
            input_data = self.__loader__(os.path.join(self.input_data_dir, str(item) + '.txt'))
            input_data = (input_data - self.input_mean) / self.input_std
            output_data = self.__loader__(os.path.join(self.output_data_dir, str(item) + '.txt'))
            output_data = (output_data - self.output_mean) / self.output_std
            self.input_data[item] = input_data
            self.output_data[item] = output_data
        else:
            input_data = self.input_data[item]
            output_data = self.output_data[item]
        return input_data, output_data
