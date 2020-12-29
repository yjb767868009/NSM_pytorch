import os
import torch.nn.utils.rnn as rnn_utils
from .seq_data_set import DataSet


def load_data(input_dir, label_dir, cache=True):
    data_source = DataSet(input_dir, label_dir, cache)
    if cache is True:
        print("Loading cache")
        data_source.load_all_data()
        print("Loading finish")
    train_source = data_source
    test_source = data_source
    return train_source, test_source


def collate_fn(data):
    batch_size = len(data)
    input_data = [data[i][0] for i in range(batch_size)]
    output_data = [data[i][1] for i in range(batch_size)]
    input_data.sort(key=lambda x: len(x), reverse=True)
    output_data.sort(key=lambda x: len(x), reverse=True)
    data_length = [len(sq) for sq in input_data]
    input_data = rnn_utils.pad_sequence(input_data, batch_first=True, padding_value=0)
    output_data = rnn_utils.pad_sequence(output_data, batch_first=True, padding_value=0)
    return [input_data, output_data], data_length
