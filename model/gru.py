import torch.nn as nn

from .base import RNNModel


class GRU(RNNModel):
    def __init__(self, **kwargs):
        rnn = self.build(**kwargs)
        kwargs["model_type"] = "GRU"
        super(GRU, self).__init__(rnn=rnn, **kwargs)

    def build(self,
              input_size,
              hidden_size,
              num_layers,
              dropout,
              batch_first=False,
              **kwargs):
        return nn.GRU(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      batch_first=batch_first)
