import os
from model.utils import load_data
from model.c_rnn_gan.base_model import BaseModel
from model.c_rnn_gan.gan_model import GANModel
from model.c_rnn_gan.config import conf


def initialize_model(name, model_config, data):
    print("Initializing model...")
    train_source, test_source = data
    model_config['train_source'] = train_source
    model_config['test_source'] = test_source
    model = eval(name)(**model_config)
    print("Model initialization complete.")
    return model


def initialization(name, cache=True):
    print("Initializing...")
    os.environ["CUDA_VISIBLE_DEVICES"] = conf["CUDA_VISIBLE_DEVICES"]
    data_root = conf['data_root']
    model = None
    if name == "base_model":
        model = initialize_model("BaseModel", conf['base_model'],
                                 load_data(os.path.join(data_root, 'Input'), os.path.join(data_root, 'Label'),
                                           cache=cache))
    if name == "gan_model":
        model = initialize_model("GANModel", conf['gan_model'],
                                 load_data(os.path.join(data_root, 'Output'), os.path.join(data_root, 'Label'),
                                           cache=cache))
    return model
