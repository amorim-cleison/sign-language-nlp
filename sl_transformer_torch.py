import math
import time
import torch
import torch.nn as nn
from commons.log import log
from commons.util import normpath

from dataset import build_dataset, build_iterator, PAD_WORD


def run(seed, cuda, config, dataset_args, model_args, training_args,
        transfer_learning_args, **kwargs):
    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        if not cuda:
            log("WARNING: You have a CUDA device, so you should \
                probably run with --cuda")

    device = torch.device("cuda" if cuda else "cpu")

    ###########################################################################
    # Load data
    ###########################################################################
    train_data, val_data, test_data = load_data(device=device,
                                                **dataset_args,
                                                **training_args)
    src_vocab = train_data.fields["src"].vocab
    trg_vocab = train_data.fields["trg"].vocab

    ###########################################################################
    # Build the model
    ###########################################################################
    model = build_model(device=device,
                        src_vocab=src_vocab,
                        tgt_vocab=trg_vocab,
                        **model_args)
    corpus = None
    criterion = nn.NLLLoss()  # FIXME: replace by the LabelSmoothing

    ###########################################################################
    # Training code
    ###########################################################################
    run_training(model=model,
                 corpus=corpus,
                 criterion=criterion,
                 train_data=train_data,
                 val_data=val_data,
                 device=device,
                 **training_args)

    run_test(corpus=corpus,
             criterion=criterion,
             test_data=test_data,
             **training_args)


def load_data(batch_size, device, **kwargs):
    return build_dataset(**kwargs)


def get_batches(data, device, batch_size, train, **kwargs):
    data_iter = build_iterator(data, batch_size, device, train)
    return data_iter


def build_model(device, N, d_model, d_ff, h, dropout, src_vocab, tgt_vocab,
                **kwargs):
    # FIXME: check if there is difference in using the torch's implementation
    # model = model.TransformerModel(ntokens, args.emsize, args.nhead,
    # args.nhid, args.nlayers, args.dropout).to(device)
    # FIXME: check parameters

    from transformer_model import PositionalEncoding

    class CustomModel(nn.Module):
        from typing import Optional
        from torch import Tensor

        def __init__(self, d_model, nhead, num_encoder_layers,
                     num_decoder_layers, dim_feedforward, dropout, src_vocab,
                     tgt_vocab, pad_word, **kwargs):
            super(CustomModel, self).__init__()
            self.d_model = d_model
            self.pad_word = pad_word
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            self.src_embedding = nn.Embedding(len(src_vocab), d_model)
            self.src_pos_encoding = PositionalEncoding(d_model, dropout)
            self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model)
            self.tgt_pos_encoding = PositionalEncoding(d_model, dropout)
            self.transformer = nn.Transformer(d_model, nhead,
                                              num_encoder_layers,
                                              num_decoder_layers,
                                              dim_feedforward, dropout)
            self.linear = nn.Linear(d_model, len(tgt_vocab))
            self.softmax = nn.functional.log_softmax

        def forward(self,
                    src: Tensor,
                    tgt: Tensor,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None):
            # Attention masks:
            src_mask = None
            tgt_mask = self.transformer.generate_square_subsequent_mask(
                tgt.size(0))

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

        def generate_padding_mask(self, data, vocab):
            pad_idx = vocab.stoi[self.pad_word]
            padding_mask = (data != pad_idx)
            return padding_mask.transpose(0, 1)

    return CustomModel(d_model=d_model,
                       nhead=h,
                       num_encoder_layers=N,
                       num_decoder_layers=N,
                       dim_feedforward=d_ff,
                       dropout=dropout,
                       src_vocab=src_vocab,
                       tgt_vocab=tgt_vocab,
                       pad_word=PAD_WORD).to(device)

    # return nn.Sequential(embedding, pos_encoder, transformer)

    # return nn.Transformer(d_model=d_model,
    #                             nhead=h,
    #                             num_encoder_layers=N,
    #                             num_decoder_layers=N,
    #                             dim_feedforward=d_ff,
    #                             dropout=dropout).to(device)

    # from .transformer_model import TransformerModel
    # return TransformerModel(d_model=d_model,
    #                         nhead=h,
    #                         num_encoder_layers=N,
    #                         src_vocab_size=src_vocab_size,
    #                         dim_feedforward=d_ff,
    #                         dropout=dropout).to(device)


def run_training(model, epochs, corpus, criterion, train_data, val_data, lr,
                 log_interval, checkpoint_dir, **kwargs):
    # Loop over epochs.
    # lr = args.lr
    best_val_loss = None
    save = normpath(f"{checkpoint_dir}/weights.pt")

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train(epoch, model, corpus, train_data, criterion, lr,
                  log_interval, **kwargs)
            val_loss = evaluate(val_data)
            log('-' * 89)
            log('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch,
                                           (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
            log('-' * 89)

            # Save the model if the validation loss is the best we've seen so
            # far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in
                # the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        log('-' * 89)
        log('Exiting from training early')


def run_test(corpus, criterion, test_data, checkpoint_dir, **kwargs):
    save = normpath(f"{checkpoint_dir}/weights.pt")

    # Load the best saved model.
    with open(save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        # if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        #     model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(model=model,
                         corpus=corpus,
                         criterion=criterion,
                         data_source=test_data)
    log('=' * 89)
    log('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    log('=' * 89)


def evaluate(model, corpus, criterion, data_source, **kwargs):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    batches = get_batches(data_source)

    with torch.no_grad():
        for i, batch in enumerate(batches):
            data, targets = batch.src, batch.tgt
            output = model(data)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train(epoch, model, corpus, train_data, criterion, lr, log_interval,
          **kwargs):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(train_data.fields["trg"].vocab)
    batches = get_batches(data=train_data, train=True, **kwargs)

    for i, batch in enumerate(batches):
        # data, targets = batch.src, batch.trg
        # Starting each batch, we detach the hidden state from how it was
        # previously produced.
        # If we didn't, the model would try backpropagating all the way to
        # start of the dataset.
        model.zero_grad()
        # output = model(data)
        output = model.forward(batch.src, batch.trg)

        # TODO: check this
        output = torch.argmax(output.transpose(0, 1), dim=-1)

        output = output.view(-1, ntokens)
        loss = criterion(output, batch.trg)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in
        # RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            log('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch,
                    len(train_data) // args.bptt,
                    lr, elapsed * 1000 / log_interval, cur_loss,
                    math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        # if args.dry_run:
        #     break
