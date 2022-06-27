from model.base import EncoderDecoderAttnBaseBkp


class EncoderDecoderGRUAttn(EncoderDecoderAttnBaseBkp):
    def __init__(self, **kwargs):
        super(EncoderDecoderGRUAttn, self).__init__(rnn_type='gru', **kwargs)
