import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
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
        self.rnn = rnn
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        src_ntoken = len(src_vocab)
        tgt_ntoken = len(tgt_vocab)

        self.drop = nn.Dropout(p=dropout)
        self.encoder = nn.Embedding(num_embeddings=src_ntoken,
                                    embedding_dim=input_size)
        self.decoder = nn.Linear(in_features=hidden_size,
                                 out_features=tgt_ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models"
        # (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for
        # Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            assert (hidden_size == input_size),\
                'When using the tied flag, `hidden_size` must be equal to '\
                '`input_size`'
            self.decoder.weight = self.encoder.weight

        self.init_weights()
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
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, bsz, self.hidden_size)
