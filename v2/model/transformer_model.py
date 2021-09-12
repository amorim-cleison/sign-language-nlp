import math

import torch.nn as nn

from .positional_encoding import PositionalEncoding


class TransformerModel(nn.Module):
    from typing import Optional

    from torch import Tensor

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout, src_ntoken, tgt_ntoken, **kwargs):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_ntoken, d_model)
        self.src_pos_encoding = PositionalEncoding(d_model, dropout)
        self.tgt_embedding = nn.Embedding(tgt_ntoken, d_model)
        self.tgt_pos_encoding = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers,
                                          num_decoder_layers, dim_feedforward,
                                          dropout)
        self.linear = nn.Linear(d_model, tgt_ntoken)
        self.softmax = nn.functional.log_softmax

    def to(self, device):
        self.src_embedding = self.src_embedding.to(device)
        self.src_pos_encoding = self.src_pos_encoding.to(device)
        self.tgt_embedding = self.tgt_embedding.to(device)
        self.tgt_pos_encoding = self.tgt_pos_encoding.to(device)
        self.transformer = self.transformer.to(device)
        self.linear = self.linear.to(device)
        return super().to(device)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        assert (src is not None), "`src` is a required paramenter"
        assert (tgt is not None), "`tgt` is a required paramenter"

        # Embeddings:
        src_embed = self.forward_embedding(src, self.src_embedding,
                                           self.src_pos_encoding)
        tgt_embed = self.forward_embedding(tgt, self.tgt_embedding,
                                           self.tgt_pos_encoding)

        # Forward:
        output = self.transformer(
            src=src_embed,
            tgt=tgt_embed,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        output = self.linear(output)
        output = self.softmax(output, dim=-1)
        return output

    def forward_embedding(self, x, embedding, pos_encoding):
        x = embedding(x) * math.sqrt(self.d_model)
        x = pos_encoding(x)
        return x
