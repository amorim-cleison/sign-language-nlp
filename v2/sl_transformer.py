import math
import time
import torch
import torch.nn as nn
from commons.log import log
from commons.util import normpath
from .dataset import build_dataset, build_iterator, PAD_WORD


def run(mode, seed, cuda, config, dataset_args, model_args, training_args,
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
    tgt_vocab = train_data.fields["tgt"].vocab

    ###########################################################################
    # Build the model
    ###########################################################################
    model = build_model(device=device,
                        src_vocab=src_vocab,
                        tgt_vocab=tgt_vocab,
                        **model_args)

    criterion = build_criterion(mode=mode,
                                tgt_vocab=tgt_vocab,
                                pad_word=PAD_WORD,
                                device=device,
                                **training_args)
    optimizer = build_optimizer(
        mode=mode,
        model=model,
        **model_args,
        **training_args,
    )

    ###########################################################################
    # Training code
    ###########################################################################
    run_training(mode=mode,
                 model=model,
                 criterion=criterion,
                 optmizer=optimizer,
                 train_data=train_data,
                 val_data=val_data,
                 device=device,
                 **training_args)

    run_test(criterion=criterion,
             test_data=test_data,
             device=device,
             **training_args)


def load_data(batch_size, device, **kwargs):
    return build_dataset(**kwargs)


def get_batches(data, device, batch_size, train, **kwargs):
    data_iter = build_iterator(data, batch_size, device, train)
    return data_iter


def build_optimizer(mode, model, d_model, warm_up, lr, betas, eps, **kwargs):
    if mode == "old":
        from .loss import NoamOpt
        adam = torch.optim.Adam(model.parameters(),
                                lr=lr,
                                betas=tuple(betas),
                                eps=eps)
        optimizer = NoamOpt(model_size=d_model,
                            factor=1,
                            warmup=warm_up,
                            optimizer=adam,
                            **kwargs)
    else:
        optimizer = None
    return optimizer


def build_criterion(mode, tgt_vocab, label_smoothing, pad_word, device,
                    **kwargs):
    if mode == "bkp":
        from .loss import LabelSmoothingLoss
        pad_idx = tgt_vocab.stoi[pad_word]
        criterion = LabelSmoothingLoss(size=len(tgt_vocab),
                                       padding_idx=pad_idx,
                                       smoothing=label_smoothing).to(device)
    else:
        criterion = nn.NLLLoss()
    return criterion


def build_model(device, N, d_model, d_ff, h, dropout, src_vocab, tgt_vocab,
                **kwargs):
    from .model import CustomModel

    model = CustomModel(d_model=d_model,
                        nhead=h,
                        num_encoder_layers=N,
                        num_decoder_layers=N,
                        dim_feedforward=d_ff,
                        dropout=dropout,
                        src_vocab=src_vocab,
                        tgt_vocab=tgt_vocab,
                        pad_word=PAD_WORD).to(device)
    return nn.DataParallel(model)


def run_training(mode, model, epochs, criterion, optmizer, train_data,
                 val_data, lr, log_interval, checkpoint_dir, **kwargs):
    # Loop over epochs.
    # lr = args.lr
    best_val_loss = None
    save = normpath(f"{checkpoint_dir}/weights.pt")

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            if mode == "old":
                train_with_optmizer(epoch, model, train_data, criterion,
                                    optmizer, lr, log_interval, **kwargs)
            else:
                train(epoch, model, train_data, criterion, lr, log_interval,
                      **kwargs)

            val_loss = evaluate(model, criterion, val_data, **kwargs)
            log('-' * 89)
            log(f'| end of epoch {epoch:3d} '
                f'| time: {(time.time() - epoch_start_time):5.2f}s '
                f'| valid loss {val_loss:5.2f} '
                f'| valid ppl {math.exp(val_loss):8.2f}')
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


def run_test(criterion, test_data, checkpoint_dir, **kwargs):
    save = normpath(f"{checkpoint_dir}/weights.pt")

    # Load the best saved model.
    with open(save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(model=model,
                         criterion=criterion,
                         data_source=test_data,
                         **kwargs)
    log('=' * 89)
    log('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    log('=' * 89)


def evaluate(model, criterion, data_source, **kwargs):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    batches = get_batches(data=data_source, train=False, **kwargs)

    with torch.no_grad():
        for i, batch in enumerate(batches):
            data, targets = batch.src, batch.tgt
            output = model.forward(data, targets)
            output = output.view(-1, output.size(-1))
            targets = targets.view(-1)
            # TODO: review this:
            # total_loss += len(data) * criterion(output, targets).item()
            total_loss += data.size(-1) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train(epoch, model, train_data, criterion, lr, log_interval, **kwargs):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    batches = get_batches(data=train_data, train=True, **kwargs)

    for i, batch in enumerate(batches):
        data, targets = batch.src, batch.tgt

        # Starting each batch, we detach the hidden state from how it was
        # previously produced.
        # If we didn't, the model would try backpropagating all the way to
        # start of the dataset.
        model.zero_grad()
        output = model.forward(data, targets)
        output = output.view(-1, output.size(-1))
        targets = targets.view(-1)
        loss = criterion(output, targets)
        loss.backward()

        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            log(f'| epoch {epoch:3d} | {i:5d}/{len(batches):5d} batches '
                f'| lr {lr:02.2f} '
                f'| ms/batch {elapsed * 1000 / log_interval:5.2f} '
                f'| loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
            total_loss = 0
            start_time = time.time()


def train_with_optmizer(epoch, model, train_data, criterion, optimizer, lr,
                        log_interval, **kwargs):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    batches = get_batches(data=train_data, train=True, **kwargs)

    for i, batch in enumerate(batches):
        data, targets = batch.src, batch.tgt

        # Starting each batch, we detach the hidden state from how it was
        # previously produced.
        # If we didn't, the model would try backpropagating all the way to
        # start of the dataset.
        # model.zero_grad()  # FIXME: Removed line
        optimizer.zero_grad()  # FIXME: New line

        output = model.forward(data, targets)
        output = output.view(-1, output.size(-1))
        targets = targets.view(-1)
        loss = criterion(output, targets)
        loss.backward()

        optimizer.step()  # FIXME: New line

        # for p in model.parameters():  # FIXME: Removed line
        #     p.data.add_(p.grad, alpha=-lr)  # FIXME: Removed line

        total_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            log(f'| epoch {epoch:3d} | {i:5d}/{len(batches):5d} batches '
                # f'| lr {lr:02.2f} ' # FIXME: Removed line
                f'| lr {optimizer.lr:02.6f} '  # FIXME: New line
                f'| ms/batch {elapsed * 1000 / log_interval:5.2f} '
                f'| loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
            total_loss = 0
            start_time = time.time()
