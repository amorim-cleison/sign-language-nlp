import torch.nn as nn
import torch

import random
import model.util as util


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 dropout, src_vocab, tgt_vocab, batch_first):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.src_vocab = src_vocab

        src_pad_idx = util.get_pad_idx(src_vocab)

        # Layers
        self.embedding = nn.Embedding(num_embeddings=input_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=src_pad_idx)
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]

        embedded = self.pack(embedded, src)  # Pack

        outputs, hidden = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden

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


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers,
                 dropout, src_vocab, tgt_vocab, batch_first):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        src_pad_idx = util.get_pad_idx(src_vocab)

        self.embedding = nn.Embedding(num_embeddings=output_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=src_pad_idx)
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           batch_first=batch_first)
        self.fc_out = nn.Linear(in_features=hidden_size,
                                out_features=self.output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(-1)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        embedded = self.pack(embedded, input)  # Pack

        output, hidden = self.rnn(embedded, hidden)
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        output = self.unpack(output)  # Unpack

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(1))
        # prediction = [batch size, output dim]

        prediction = self.softmax(prediction)

        return prediction, hidden

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


class Seq2SeqLSTM(nn.Module):
    # def __init__(self, encoder, decoder, device):
    def __init__(self, embedding_size, hidden_size, num_layers, dropout,
                 src_vocab, tgt_vocab, batch_first, device):
        super(Seq2SeqLSTM, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.batch_first = batch_first

        input_size = len(src_vocab)
        output_size = len(tgt_vocab)

        # self.encoder = encoder
        # self.decoder = decoder
        self.encoder = Encoder(input_size, embedding_size, hidden_size,
                               num_layers, dropout, src_vocab, tgt_vocab,
                               batch_first)
        self.decoder = Decoder(output_size, embedding_size, hidden_size,
                               num_layers, dropout, src_vocab, tgt_vocab,
                               batch_first)
        self.device = device

        # assert self.encoder.hidden_size == self.decoder.hidden_size, \
        #     "Hidden dimensions of encoder and decoder must be equal!"
        # assert self.encoder.num_layers == self.decoder.num_layers, \
        #     "Encoder and decoder must have equal number of layers!"

        self.apply(self.init_weights)

    def forward(self, X, y, teacher_forcing_ratio=0.5, **kwargs):
        src, tgt = X, y

        tgt = tgt if tgt.ndim > 1 else tgt.unsqueeze(-1)

        # X = [X len, batch size]
        # y = [y len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = self.get_batch_size(src)  # y.shape[1]
        tgt_len = tgt.shape[1]  #y.shape[0]
        tgt_vocab_size = len(self.tgt_vocab)  # self.decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len,
                              tgt_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        # input = y  # y[0, :]
        from dataset.constant import BOS_WORD
        BOS_IDX = self.tgt_vocab.stoi[BOS_WORD]
        BOS_TENSOR = torch.full((batch_size, ), BOS_IDX).to(self.device)
        input = BOS_TENSOR

        for t in range(0, tgt_len):  # range(1, tgt_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(input, hidden)

            # place predictions in a tensor holding predictions for each token
            outputs[:, t] = output
            # outputs = output

            # # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # # get the highest predicted token from our predictions
            top1 = output.argmax(dim=-1)

            # # if teacher forcing, use actual next token as next input
            # # if not, use predicted token
            input = tgt[:, t] if teacher_force else top1

        # return outputs
        return self.get_last_time_step(outputs)

    def init_weights(self, model):
        for name, param in model.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_batch_size(self, X):
        return X.size(0) if self.batch_first else X.size(1)

    def get_last_time_step(self, output):
        return output[:, -1, :] if self.batch_first else output[-1, :, :]
