import os

from .seq_data_set import DataSet


def load_data(input_dir, label_dir, cache=True):
    data_source = DataSet(input_dir, label_dir, cache)
    if cache:
        data_source.load_all_data()
    train_source = data_source
    test_source = data_source
    return train_source, test_source
