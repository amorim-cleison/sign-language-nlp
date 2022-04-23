import torch.nn as nn
import torch
import model.util as util


class LSTMNew(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 src_vocab,
                 tgt_vocab,
                 dropout=0.5,
                 batch_first=False,
                 **kwargs):
        super(LSTMNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        src_pad_idx = util.get_pad_idx(src_vocab)

        # Layers
        self.embedding = nn.Embedding(num_embeddings=len(src_vocab),
                                      embedding_dim=input_size,
                                      padding_idx=src_pad_idx)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=len(tgt_vocab))
        self.softmax = nn.LogSoftmax(dim=-1)

        # Weights
        self.apply(self.init_weights)

    def to(self, device):
        self.embedding = self.embedding.to(device)
        self.lstm = self.lstm.to(device)
        self.drop = self.drop.to(device)
        self.linear = self.linear.to(device)
        self.softmax = self.softmax.to(device)
        self.device = device
        return super().to(device)

    def forward(self, X, lengths, **kwargs):
        hidden = self.init_hidden(self.get_batch_size(X))

        embedded = self.embedding(X)
        output = self.drop(embedded)

        output = self.pack(output, X, lengths)  # Pack

        output, hidden = self.lstm(output, hidden)

        output = self.unpack(output)  # Unpack

        output = self.get_last_time_step(output)
        output = self.linear(output)
        output = self.softmax(output)

        return output

    def init_weights(self, model):
        for name, param in model.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers,
                            batch_size,
                            self.hidden_size,
                            device=self.device),
                torch.zeros(self.num_layers,
                            batch_size,
                            self.hidden_size,
                            device=self.device))

    def get_batch_size(self, X):
        return X.size(0) if self.batch_first else X.size(1)

    def get_last_time_step(self, output):
        return output[:, -1, :] if self.batch_first else output[-1, :, :]

    def pack(self, output, X=None, lengths=None):
        import torch.nn.utils.rnn as t

        assert (X is not None) or (
            output is not None), "One of `X` or `output` should be informed."

        if lengths is None:
            lengths = util.resolve_lengths(X, self.src_vocab)

        return t.pack_padded_sequence(input=output,
                                      lengths=lengths.cpu(),
                                      batch_first=self.batch_first,
                                      enforce_sorted=False)

    def unpack(self, output):
        import torch.nn.utils.rnn as t

        unpacked, _ = t.pad_packed_sequence(sequence=output,
                                            batch_first=self.batch_first)
        return unpacked
