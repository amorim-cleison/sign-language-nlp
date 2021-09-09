import math
import time

import torch
import torch.nn as nn
from commons.log import log
from commons.util import normpath

from .dataset import build_dataset, build_iterator, PAD_WORD
from .util import (generate_mask, generate_padding_mask, log_step, log_model,
                   log_data, save_eval_outputs, save_step)
"""
This code was based on the link:
- https://pytorch.org/tutorials/beginner/transformer_tutorial.html

Other links:
- https://github.com/pytorch/examples/tree/master/word_language_model
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""


def run(seed, cuda, config, dataset_args, model_args, training_args,
        transfer_learning_args, **kwargs):
    # FIXME: externalize this to config
    model_type = "LSTM"

    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)

    ###########################################################################
    # Prepare CUDA
    ###########################################################################
    if torch.cuda.is_available():
        if not cuda:
            log("WARNING: You have a CUDA device, so you should probably "
                "run with --cuda")

    device = torch.device("cuda" if cuda else "cpu")

    ###########################################################################
    # Load data
    ###########################################################################
    train_data, val_data, test_data = load_data(device=device,
                                                **dataset_args,
                                                **training_args)
    src_vocab, tgt_vocab, _ = get_vocabs(train_data)

    log_data({
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "src vocab": src_vocab,
        "tgt vocab": tgt_vocab
    })

    ###########################################################################
    # Build the model
    ###########################################################################
    model = build_model(device=device,
                        model_type=model_type,
                        src_vocab=src_vocab,
                        tgt_vocab=tgt_vocab,
                        **model_args)

    criterion = build_criterion(**training_args)
    optimizer = build_optimizer(model=model, **training_args)
    scheduler = build_scheduler(optimizer=optimizer, **training_args)

    log_model({
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler
    })

    ###########################################################################
    # Training code
    ###########################################################################
    run_training(model=model,
                 model_type=model_type,
                 criterion=criterion,
                 optimizer=optimizer,
                 scheduler=scheduler,
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


def get_vocabs(dataset):
    return dataset.fields["src"].vocab,\
           dataset.fields["tgt"].vocab,\
           dataset.fields["file"].vocab


def get_batches(data, device, batch_size, train, **kwargs):
    return build_iterator(data, batch_size, device, train)


def build_optimizer(model, lr, **kwargs):
    # return torch.optim.Adam(model.parameters(),
    #                         lr=lr,
    #                         betas=tuple(betas),
    #                         eps=eps)
    return torch.optim.SGD(model.parameters(), lr=lr)


def build_scheduler(optimizer, lr_step_size, lr_step_gamma, **kwargs):
    return torch.optim.lr_scheduler.StepLR(optimizer,
                                           step_size=lr_step_size,
                                           gamma=lr_step_gamma)


def build_criterion(**kwargs):
    # return nn.NLLLoss()
    return nn.CrossEntropyLoss(ignore_index=1)


def build_model(device, model_type, N, d_model, d_ff, h, dropout, src_vocab,
                tgt_vocab, **kwargs):
    def to_parallel(model, device):
        return nn.DataParallel(model) if (device.type == "cuda") else model

    if model_type == "Transformer":
        from .model import TransformerModel
        model = TransformerModel(d_model=d_model,
                                 nhead=h,
                                 num_encoder_layers=N,
                                 num_decoder_layers=N,
                                 dim_feedforward=d_ff,
                                 dropout=dropout,
                                 src_ntoken=len(src_vocab),
                                 tgt_ntoken=len(tgt_vocab)).to(device)
    else:
        args = {
            "rnn_type": model_type,
            "src_ntoken": len(src_vocab),
            "tgt_ntoken": len(tgt_vocab),
            "ninp": d_model,
            "nhid": d_ff,
            "nlayers": N,
            "dropout": dropout,
            "tie_weights": False
        }
        from .model import RNNModel
        model = RNNModel(**args).to(device)
    return to_parallel(model, device)


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def run_training(model, epochs, criterion, optimizer, scheduler, train_data,
                 val_data, log_interval, checkpoint_dir, **kwargs):
    best_val_loss = float("inf")
    save = normpath(f"{checkpoint_dir}/weights.pt")

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train(epoch=epoch,
                  model=model,
                  data_source=train_data,
                  criterion=criterion,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  log_interval=log_interval,
                  checkpoint_dir=checkpoint_dir,
                  **kwargs)

            val_loss, val_acc = evaluate(epoch=epoch,
                                         model=model,
                                         criterion=criterion,
                                         data_source=val_data,
                                         checkpoint_dir=checkpoint_dir,
                                         **kwargs)

            step_data = {
                "epoch": f"{epoch:3d}",
                "time": f"{(time.time() - epoch_start_time):5.2f}s",
                "acc": f"{val_acc:5.2f}",
                "loss": f"{val_loss:5.2f}",
                "ppl": f"{math.exp(val_loss):8.2f}"
            }
            log_step("valid", sep="-", **step_data)
            save_step("valid", checkpoint_dir, **step_data)

            # Save the model if the validation loss is the best we've seen so
            # far.
            if val_loss < best_val_loss:
                with open(save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss

            scheduler.step()
    except KeyboardInterrupt:
        log('-' * 100)
        log('Exiting from training early')


def run_test(model_type, criterion, test_data, checkpoint_dir, **kwargs):
    save = normpath(f"{checkpoint_dir}/weights.pt")

    # Load the best saved model.
    with open(save, 'rb') as f:
        model = torch.load(f)

        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if model_type in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

    # Run on test data.
    test_loss, test_acc = evaluate(model=model,
                                   criterion=criterion,
                                   data_source=test_data,
                                   checkpoint_dir=checkpoint_dir,
                                   **kwargs)
    step_data = {
        "acc": f"{test_acc:5.2f}",
        "loss": f"{test_loss:5.2f}",
        "ppl": f"{math.exp(test_loss):8.2f}"
    }
    log_step("test", sep="=", **step_data)
    save_step("test", checkpoint_dir, **step_data)


def evaluate(model,
             model_type,
             criterion,
             data_source,
             device,
             checkpoint_dir,
             batch_size,
             epoch=None,
             **kwargs):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total_acc = 0
    src_vocab, tgt_vocab, file_vocab = get_vocabs(data_source)
    batches = get_batches(data=data_source,
                          train=False,
                          device=device,
                          batch_size=batch_size,
                          **kwargs)

    if model_type != 'Transformer':
        hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for i, batch in enumerate(batches):
            # Data:
            src, tgt, files = batch.src, batch.tgt, batch.file

            # Forward:
            if model_type == 'Transformer':
                src_mask = None
                tgt_mask = generate_mask(tgt, model).to(device)
                src_padding_mask = \
                    generate_padding_mask(src, src_vocab).to(device)
                tgt_padding_mask = \
                    generate_padding_mask(tgt, tgt_vocab).to(device)

                output = model.forward(src=src,
                                       tgt=tgt,
                                       src_mask=src_mask,
                                       tgt_mask=tgt_mask,
                                       src_key_padding_mask=src_padding_mask,
                                       tgt_key_padding_mask=tgt_padding_mask)
            else:
                hidden = model.init_hidden(len(batch))  # FIXME
                output, hidden = model.forward(input=src, hidden=hidden)
                # hidden = repackage_hidden(hidden)

            # Loss:
            total_loss += calc_loss(criterion, output, tgt).item()

            # Accuracy:
            total_acc += calc_accuracy(output, tgt).item()

            # Save outputs:
            output = output[-1]
            tgt = tgt[-1]
            files = files[-1]
            save_eval_outputs(output, tgt, files, tgt_vocab, file_vocab,
                              checkpoint_dir, epoch)

    loss = total_loss / len(batches)
    accuracy = total_acc / len(batches)
    return loss, accuracy


def train(epoch, model, model_type, data_source, criterion, optimizer,
          scheduler, log_interval, checkpoint_dir, device, batch_size,
          **kwargs):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    src_vocab, tgt_vocab, _ = get_vocabs(data_source)
    batches = get_batches(data=data_source,
                          train=True,
                          device=device,
                          batch_size=batch_size,
                          **kwargs)

    if model_type != 'Transformer':
        hidden = model.init_hidden(batch_size)

    for i, batch in enumerate(batches):
        # Data:
        src, tgt = batch.src, batch.tgt

        # Forward:
        optimizer.zero_grad()

        if model_type == 'Transformer':
            src_mask = None
            tgt_mask = generate_mask(tgt, model).to(device)
            src_padding_mask = generate_padding_mask(src, src_vocab).to(device)
            tgt_padding_mask = generate_padding_mask(tgt, tgt_vocab).to(device)

            output = model.forward(src=src,
                                   tgt=tgt,
                                   src_mask=src_mask,
                                   tgt_mask=tgt_mask,
                                   src_key_padding_mask=src_padding_mask,
                                   tgt_key_padding_mask=tgt_padding_mask)
        else:
            # hidden = repackage_hidden(hidden)
            hidden = model.init_hidden(len(batch))  # FIXME
            output, hidden = model.forward(input=src, hidden=hidden)

        # Loss and optimization:
        loss = calc_loss(criterion, output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            step_data = {
                "epoch": f"{epoch:3d}",
                "batch": f"{i:5d} /{len(batches):5d}",
                "lr": f"{scheduler.get_last_lr()[0]:02.5f}",
                "ms/batch": f"{elapsed * 1000 / log_interval:5.2f}",
                "loss": f"{cur_loss:5.2f}",
                "ppl": f"{math.exp(cur_loss):8.2f}"
            }
            log_step("train", **step_data)
            # save_step("train", checkpoint_dir, **step_data)
            total_loss = 0
            start_time = time.time()


def calc_loss(criterion, output, targets):
    # total_loss += len(data) * criterion(output, targets).item()
    # output = output.view(-1, output.size(-1))
    # targets = targets.view(-1)
    output = output[-1]
    targets = targets[-1]
    return criterion(output, targets)


def calc_accuracy(output, targets):
    # As the targets are shifted right (the first index is BOS token), we will
    # consider since the second index:
    output = output[1:]
    targets = targets[1:]

    output_labels = torch.argmax(output, dim=-1)
    corrects = (output_labels == targets)
    total = targets.nelement()
    return corrects.sum() / total
