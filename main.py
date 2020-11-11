import argparse

from config import conf
from model.initialization import initialization

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--cache', default=True, help='cache: if set as TRUE all the training data will be loaded at once'
                                                  ' before the training start. Default: TRUE')
parser.add_argument('--train_NSM', default=True, help='train NSM generate model')
parser.add_argument('--train_GAN', default=True, help='train Refiner GAN')
parser.add_argument('--test', default=False, help='if test model')
opt = parser.parse_args()

if __name__ == '__main__':
    model = initialization(train=True)
    if opt.test:
        print("Testing START")
        model.test()
        print("Testing COMPLETE")
    else:
        if opt.train_NSM:
            model.train()
        if opt.train_GAN:
            model.train_gan()
