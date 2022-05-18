from model.base import EncoderDecoderAttnBaseBkp


class EncoderDecoderLSTMAttn(EncoderDecoderAttnBaseBkp):
    def __init__(self, **kwargs):
        super(EncoderDecoderLSTMAttn, self).__init__(rnn_type="lstm", **kwargs)
