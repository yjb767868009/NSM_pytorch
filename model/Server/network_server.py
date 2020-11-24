from model.c_rnn_gan import conf
from model.utils.initialization import initialize_model


class Server(object):
    def __init__(self):
        self.base_model = initialize_model("BaseModel", conf['base_model'], (None, None))
        self.gan_model = initialize_model("GANModel", conf['gan_model'], (None, None))
        self.base_model.load_param()
        self.gan_model.load_param()

    def forward(self, x):
        x_length = len(x)
        x = self.base_model.forward(x, x_length)
        x = self.gan_model.forward(x)
        return x
