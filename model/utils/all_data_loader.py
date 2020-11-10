import numpy as np
import os

from .all_data_set import DataSet


def normalize(X, N, save_path=None):
    mean = N[0]
    std = N[1]
    for i in range(std.size):
        if std[i] == 0:
            std[i] = 1
    if save_path is not None:
        mean.tofile(save_path + 'mean.bin')
        std.tofile(save_path + 'std.bin')
    return (X - mean) / std, mean, std


def load_data(dataset_path, save_path, cache=True):
    input_data, input_mean, input_std = normalize(
        np.float32(np.loadtxt(dataset_path + '/Input.txt')),
        np.float32(np.loadtxt(dataset_path + '/InputNorm.txt')),
        save_path=save_path + '/X')
    output_data, output_mean, output_std = normalize(
        np.float32(np.loadtxt(dataset_path + '/Output.txt')),
        np.float32(np.loadtxt(dataset_path + '/OutputNorm.txt')),
        save_path=save_path + '/Y')
    data_source = DataSet(input_data, input_data.shape[1], input_mean, input_std,
                          output_data, output_data.shape[1], output_mean, output_std,
                          input_data.shape[0], cache)
    train_source = data_source
    test_source = data_source
    return train_source, test_source
