import argparse

import torch.utils
import torch.utils.cpp_extension

from model.utils.data_preprocess import data_preprocess
from model.utils.initialization import initialization

# Check GPU available
if torch.cuda.is_available():
    print("CUDA_HOME:", torch.utils.cpp_extension.CUDA_HOME)
    print("torch cuda version:", torch.version.cuda)
    print("cuda is available:", torch.cuda.is_available())

parser = argparse.ArgumentParser(description="Train")
parser.add_argument("--cache", help="cache: if set as TRUE all the training data will be loaded at once"
                                    " before the training start.", action="store_true")
parser.add_argument("--data_preprocess", default="", help="if need preprocess dataset input and output")
parser.add_argument("--train_Base", help="train nsm generate model", action="store_true")
parser.add_argument("--train_GAN", help="train Refiner GAN", action="store_true")
parser.add_argument("--test", help="test model", action="store_true")
opt = parser.parse_args()

if __name__ == "__main__":
    if opt.data_preprocess != "":
        data_preprocess(opt.data_preprocess)
    if opt.train_Base is True:
        base_model = initialization("base_model", cache=opt.cache)
        base_model.train()
        if opt.test:
            base_model.test()
    if opt.train_GAN is True:
        gan_model = initialization("gan_model", cache=opt.cache)
        gan_model.train()
        if opt.test:
            gan_model.test()
