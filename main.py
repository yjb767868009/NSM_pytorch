import argparse

from model.utils.data_preprocess import data_preprocess
from model.utils.initialization import initialization

import torch.utils
import torch.utils.cpp_extension

# Check GPU available
if torch.cuda.is_available():
    print('CUDA_HOME:', torch.utils.cpp_extension.CUDA_HOME)
    print('torch cuda version:', torch.version.cuda)
    print('cuda is available:', torch.cuda.is_available())

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--cache', default=True, help='cache: if set as TRUE all the training data will be loaded at once'
                                                  ' before the training start. Default: TRUE')
parser.add_argument('--data_preprocess', default=True, help='preprocess dataset input and output')
parser.add_argument('--train_Base', default=True, help='train nsm generate model')
parser.add_argument('--train_GAN', default=True, help='train Refiner GAN')
parser.add_argument('--test', default=False, help='test model')
opt = parser.parse_args()

if __name__ == '__main__':
    if opt.data_preprocess:
        data_preprocess("E:/AI4Animation-master/AI4Animation/SIGGRAPH_Asia_2019/data")
    if opt.train_Base:
        base_model = initialization("base_model", cache=True)
        base_model.train()
        if opt.test:
            base_model.test()
    if opt.train_GAN:
        gan_model = initialization("gan_model", cache=True)
        gan_model.train()
        if opt.test:
            gan_model.test()
