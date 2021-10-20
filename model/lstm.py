import torch.nn as nn

from .base import RNNModel


class LSTM(RNNModel):
    def __init__(self, **kwargs):
        rnn = self.build(**kwargs)
        kwargs["model_type"] = "LSTM"
        super(LSTM, self).__init__(rnn=rnn, **kwargs)

    def build(self,
              input_size,
              hidden_size,
              num_layers,
              dropout,
              batch_first=False,
              **kwargs):
        return nn.LSTM(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       batch_first=batch_first)

    def _forward_rnn(self, input, hidden):
        _output, (ht, ct) = self.rnn(input, hidden)
        return ht[-1]

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        dims = (self.num_layers, batch_size, self.hidden_size)

        if self.batch_first:
            dims = dims.permute(1, 0)

        return (weight.new_zeros(*dims).to(self.device),
                weight.new_zeros(*dims).to(self.device))
