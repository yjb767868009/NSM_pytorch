import argparse

from config import conf
from model.initialization import initialization

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--cache', default=True, help='cache: if set as TRUE all the training data will be loaded at once'
                                                  ' before the training start. Default: TRUE')
opt = parser.parse_args()

if __name__ == '__main__':
    model = initialization(conf, train=opt.cache)[0]
    print("Training START")
    model.train()
    print("Training COMPLETE")
