import torch.nn as nn

from model.base import EncoderDecoderAttnBase


class EncoderDecoderGRUAttn(EncoderDecoderAttnBase):
    def __init__(self, **kwargs):
        super(EncoderDecoderGRUAttn, self).__init__(rnn_class=nn.GRU, **kwargs)
