from .seq_data_set import DataSet


def load_data(dataset_path, cache=True):
    data_source = DataSet(dataset_path, cache)
    train_source = data_source
    test_source = data_source
    return train_source, test_source
