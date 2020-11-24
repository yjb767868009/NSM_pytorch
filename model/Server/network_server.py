from model.utils.initialization import initialization


class Server(object):
    def __init__(self):
        self.base_model = initialization("base_model")
        self.gan_model = initialization("gan_model")
        self.base_model.load_param()
        self.gan_model.load_param()

    def forward(self, x):
        x_length = len(x)
        x = self.base_model.forward(x, x_length)
        x = self.gan_model.forward(x)
        return x
