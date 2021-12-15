import torch.nn as nn
import model.util as util
import torch.nn.utils.rnn as t


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a linear."""
    def __init__(self,
                 rnn,
                 model_type,
                 input_size,
                 hidden_size,
                 num_layers,
                 src_vocab,
                 tgt_vocab,
                 dropout=0.5,
                 tie_weights=False,
                 batch_first=False,
                 **kwargs):
        super(RNNModel, self).__init__()
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_ntoken = len(src_vocab)
        self.tgt_ntoken = len(tgt_vocab)
        self.batch_first = batch_first
        src_pad_idx = util.get_pad_idx(src_vocab)

        # Layers:
        self.encoder = nn.Embedding(num_embeddings=self.src_ntoken,
                                    embedding_dim=self.input_size,
                                    padding_idx=src_pad_idx)
        self.drop = nn.Dropout(p=dropout)
        self.rnn = rnn
        self.linear = nn.Linear(in_features=self.hidden_size,
                                out_features=self.tgt_ntoken)
        self.softmax = nn.functional.log_softmax

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models"
        # (Press & Wolf 2016) https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for
        # Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            assert (self.hidden_size == self.input_size),\
                'When using the tied flag, `hidden_size` must be equal to '\
                '`input_size`'
            self.linear.weight = self.encoder.weight
        self.init_weights()

    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.drop = self.drop.to(device)
        self.rnn = self.rnn.to(device)
        self.linear = self.linear.to(device)
        self.device = device
        return super().to(device)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.linear.bias)
        nn.init.uniform_(self.linear.weight, -initrange, initrange)

    def forward(self, X, lengths=None, **kwargs):
        if lengths is None:
            lengths = util.resolve_lengths(X, self.src_vocab)

        # Hidden:
        batch_size = X.size(0) if self.batch_first else X.size(1)
        hidden = self.init_hidden(batch_size)

        # Embedding:
        output = self.encoder(X)

        # Pack:
        output = t.pack_padded_sequence(input=output,
                                        lengths=lengths.cpu(),
                                        batch_first=self.batch_first,
                                        enforce_sorted=False)

        # Forward:
        output = self._forward_rnn(output, hidden)

        # Unpack:
        output, _ = t.pad_packed_sequence(sequence=output,
                                          batch_first=self.batch_first)
        output = output[:, -1, :] if self.batch_first else output[-1, :, :]

        # Other layers:
        output = self.drop(output)
        output = self.linear(output)
        output = self.softmax(output, dim=-1)
        return output

    def _forward_rnn(self, input, hidden):
        output, _ = self.rnn(input, hidden)
        return output

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        dims = (self.num_layers, batch_size, self.hidden_size)
        return weight.new_zeros(*dims).to(self.device)
