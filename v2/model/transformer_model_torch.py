import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import PositionalEncoding


class TransformerModelTorch(nn.Module):
    # def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
    def __init__(self,
                 src_ntoken,
                 tgt_ntoken,
                 d_model,
                 nhead,
                 dim_feedforward,
                 num_encoder_layers,
                 dropout=0.5,
                 **kwargs):
        super(TransformerModelTorch, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead,
                                                 dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_encoder_layers)
        self.encoder = nn.Embedding(src_ntoken, d_model)
        self.ninp = d_model
        self.decoder = nn.Linear(d_model, tgt_ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, **kwargs):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
