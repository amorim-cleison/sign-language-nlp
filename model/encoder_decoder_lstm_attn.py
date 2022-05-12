import torch.nn as nn

from model.base import EncoderDecoderAttnBase


class EncoderDecoderLSTMAttn(EncoderDecoderAttnBase):
    def __init__(self, **kwargs):
        super(EncoderDecoderLSTMAttn, self).__init__(rnn_class=nn.LSTM,
                                                    **kwargs)
