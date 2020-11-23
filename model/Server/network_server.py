from model.utils.initialization import initialize_model
from model.c_rnn_gan.config import conf


class Server(object):
    def __init__(self):
        self.model = initialize_model()

    def eval(self, x):
        return x
