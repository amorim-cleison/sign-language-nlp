import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CustomModel(nn.Module):
    from typing import Optional
    from torch import Tensor

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout, src_vocab, tgt_vocab, pad_word,
                 **kwargs):
        super(CustomModel, self).__init__()
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
        return mask

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
        return mask
