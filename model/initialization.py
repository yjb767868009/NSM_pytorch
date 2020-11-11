import os
from copy import deepcopy
import numpy as np
from .utils import load_data
from .c_rnn_gan.model import Model
from .c_rnn_gan.config import conf


def initialize_model(config, train_source, test_source):
    print("Initializing model...")
    model_config = config['model']
    model_config['save_path'] = config['save_path']
    model_config['load_path'] = config['load_path']
    model_config['train_source'] = train_source
    model_config['test_source'] = test_source
    model = Model(**model_config)
    print("Model initialization complete.")
    return model


def initialization(train=False, test=False):
    print("Initializing...")
    os.environ["CUDA_VISIBLE_DEVICES"] = conf["CUDA_VISIBLE_DEVICES"]
    print("Initializing data source...")
    train_source, test_source = load_data(**conf['data'], cache=(train or test))
    print("Data initialization complete.")
    return initialize_model(conf, train_source, test_source)
