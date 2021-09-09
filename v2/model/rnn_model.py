import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self,
                 rnn_type,
                 ninp,
                 nhid,
                 nlayers,
                 src_ntoken,
                 tgt_ntoken,
                 dropout=0.5,
                 tie_weights=False):
        super(RNNModel, self).__init__()
        self.src_ntoken = src_ntoken
        self.tgt_ntoken = tgt_ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(src_ntoken, ninp)
        self.model_type = rnn_type

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp,
                                             nhid,
                                             nlayers,
                                             dropout=dropout)
        else:
            try:
                nonlinearity = {
                    'RNN_TANH': 'tanh',
                    'RNN_RELU': 'relu'
                }[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(ninp,
                              nhid,
                              nlayers,
                              nonlinearity=nonlinearity,
                              dropout=dropout)
        self.decoder = nn.Linear(nhid, tgt_ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.sigmoid = nn.Sigmoid()

    def to(self, device):
        self.device = device
        return super().to(device)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        from torch.nn.utils.rnn import\
            (pack_padded_sequence, pad_packed_sequence)

        emb = self.drop(self.encoder(input))


        # lengths = (~input.eq(1)).sum(dim=0)
        # emb = pack_padded_sequence(emb, lengths)
        output, hidden = self.rnn(emb, hidden)
        # output, _ = pad_packed_sequence(output)


        output = self.drop(output)
        decoded = self.decoder(output)
        # decoded = decoded.view(-1, self.tgt_ntoken)
        return F.log_softmax(decoded, dim=1), hidden

        # IMPL 1: --------------------------------------
        # emb = self.drop(self.encoder(input))
        # output, hidden = self.rnn(emb, hidden)
        # output = self.drop(output)
        # decoded = self.decoder(output)
        # # decoded = decoded.view(-1, self.tgt_ntoken)
        # return F.log_softmax(decoded, dim=1), hidden

        # IMPL 2: --------------------------------------
        # x = input

        # batch_size = x.size(-1)
        # x = x.long()
        # embeds = self.encoder(x)
        # lstm_out, hidden = self.rnn(embeds, hidden)
        # lstm_out = lstm_out.contiguous().view(-1, self.nhid)

        # out = self.drop(lstm_out)
        # out = self.decoder(out)
        # out = self.sigmoid(out)

        # out = out.view(batch_size, -1)
        # out = out[:, -1]
        # return out, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz,
                                     self.nhid).to(self.device),
                    weight.new_zeros(self.nlayers, bsz,
                                     self.nhid).to(self.device))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
