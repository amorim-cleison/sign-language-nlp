import math
import torch.nn as nn
from .positional_encoding import PositionalEncoding


class CustomModel(nn.Module):
    from typing import Optional
    from torch import Tensor

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout, src_vocab, tgt_vocab, pad_word,
                 device, **kwargs):
        super(CustomModel, self).__init__()
        self.device = device
        self.d_model = d_model
        self.pad_word = pad_word
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_embedding = nn.Embedding(len(src_vocab), d_model)
        self.src_pos_encoding = PositionalEncoding(d_model, dropout)
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model)
        self.tgt_pos_encoding = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers,
                                          num_decoder_layers, dim_feedforward,
                                          dropout)
        self.linear = nn.Linear(d_model, len(tgt_vocab))
        self.softmax = nn.functional.log_softmax
        self.to(device)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        # Attention masks:
        src_mask = None
        tgt_mask = self.generate_mask(tgt)

        # Padding masks:
        src_padding_mask = self.generate_padding_mask(src, self.src_vocab)
        tgt_padding_mask = self.generate_padding_mask(tgt, self.tgt_vocab)

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
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        output = self.linear(output)
        output = self.softmax(output, dim=-1)
        return output

    def forward_embedding(self, x, embedding, pos_encoding):
        x = embedding(x) * math.sqrt(self.d_model)
        x = pos_encoding(x)
        return x

    def generate_mask(self, data):
        """
        Mask ensures that position i is allowed to attend the unmasked
        positions. If a ByteTensor is provided, the non-zero positions are
        not allowed to attend while the zero positions will be unchanged.
        If a BoolTensor is provided, positions with ``True`` are not
        allowed to attend while ``False`` values will be unchanged.
        If a FloatTensor is provided, it will be added to the attention
        weight.
        """
        mask = self.transformer.generate_square_subsequent_mask(data.size(0))
        mask = (mask != float(0.0)).bool()
        return mask.to(self.device)

    def generate_padding_mask(self, data, vocab):
        """
        Padding mask provides specified elements in the key to be ignored
        by the attention. If a ByteTensor is provided, the non-zero
        positions will be ignored while the zero positions will be
        unchanged. If a BoolTensor is provided, the positions with the
        value of ``True`` will be ignored while the position with the
        value of ``False`` will be unchanged.
        """
        pad_idx = vocab.stoi[self.pad_word]
        mask = (data == pad_idx).transpose(0, 1).bool()
        return mask.to(self.device)
