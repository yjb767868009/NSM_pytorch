import torch

from model.c_rnn_gan import conf
from model.utils.initialization import initialize_model


class Server(object):
    def __init__(self):
        self.base_model = initialize_model("BaseModel", conf['base_model'], (None, None))
        self.gan_model = initialize_model("GANModel", conf['gan_model'], (None, None))
        self.base_model.load_param()
        self.gan_model.load_param()
        self.data = torch.empty(0, 5307)
        self.full = False

    def forward(self, x):
        l = [float(a) for a in x.split(' ')]
        x = torch.Tensor([l])
        data = torch.cat((self.data, x), 0)
        if self.full is True:
            data = data[1:]
            data_length = 100
        else:
            data_length = data.size(0)
            if data_length == 100:
                self.full = True
        data = self.base_model.forward(data, data_length)
        data = self.gan_model.forward(data)
        return data
