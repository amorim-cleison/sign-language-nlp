import torch.nn as nn

from .base import RNNModel


class RNNTanh(RNNModel):
    def __init__(self, **kwargs):
        rnn = self.build(**kwargs)
        super(RNNTanh, self).__init__(rnn=rnn, model_type="RNN_TANH", **kwargs)

    def build(self, input_size, hidden_size, num_layers, dropout, **kwargs):
        return nn.RNN(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      nonlinearity="tanh",
                      dropout=dropout)
