import torch.nn as nn

from .base import RNNModel


class RNNTanh(RNNModel):
    def __init__(self, **kwargs):
        rnn = self.build(**kwargs)
        kwargs["model_type"] = "RNN_TANH"
        super(RNNTanh, self).__init__(rnn=rnn, **kwargs)

    def build(self,
              input_size,
              hidden_size,
              num_layers,
              dropout,
              batch_first=False,
              **kwargs):
        return nn.RNN(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      nonlinearity="tanh",
                      batch_first=batch_first)
