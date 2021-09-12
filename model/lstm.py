import torch.nn as nn

from .base import RNNModel


class LSTM(RNNModel):
    def __init__(self, **kwargs):
        rnn = self.build(**kwargs)
        super(LSTM, self).__init__(rnn=rnn, model_type="LSTM", **kwargs)

    def build(self, input_size, hidden_size, num_layers, dropout, **kwargs):
        return nn.LSTM(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       dropout=dropout)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz,
                                 self.hidden_size).to(self.device),
                weight.new_zeros(self.num_layers, bsz,
                                 self.hidden_size).to(self.device))
