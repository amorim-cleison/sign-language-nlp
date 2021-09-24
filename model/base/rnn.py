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
        self.device = device
        return super().to(device)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.linear.bias)
        nn.init.uniform_(self.linear.weight, -initrange, initrange)

    def forward(self, input, lengths, hidden, **kwargs):
        output = self.encoder(input)
        output = self.drop(output)

        # Pack:
        output = t.pack_padded_sequence(input=output,
                                        lengths=lengths,
                                        batch_first=False,
                                        enforce_sorted=False)

        # Forward:
        output, (ht, ct) = self.rnn(output, hidden)
        # output, (ht, ct) = self.rnn(output)

        output = self.linear(ht[-1])
        output = self.softmax(output, dim=-1)
        return output, (ht, ct)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
