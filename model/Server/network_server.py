import torch

from model.c_rnn_gan import conf
from model.utils.initialization import initialize_model


class Server(object):
    def __init__(self):
        self.base_model = initialize_model("BaseModel", conf['base_model'], (None, None))
        self.gan_model = initialize_model("GANModel", conf['gan_model'], (None, None))
        self.base_model.load_param()
        self.gan_model.load_param()

    def forward(self, x):
        lines = x.split('\n')
        data = []
        for line in lines:
            if line == "":
                break
            data_line = [float(a) for a in line.split(' ')]
            data.append(data_line)
        data = torch.Tensor(data)
        data_length = len(data)
        data = self.base_model.forward(data, data_length)
        data = self.gan_model.forward(data)
        return data
