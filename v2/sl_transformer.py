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
            log("WARNING: You have a CUDA device, so you should probably "
                "run with --cuda")

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

    criterion = build_criterion(**training_args)
    optimizer = build_optimizer(model=model, **training_args)
    scheduler = build_scheduler(optimizer=optimizer, **training_args)

    ###########################################################################
    # Training code
    ###########################################################################
    run_training(model=model,
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


def get_batches(data, device, batch_size, train, **kwargs):
    data_iter = build_iterator(data, batch_size, device, train)
    return data_iter


def build_optimizer(model, lr, betas, eps, **kwargs):
    return torch.optim.SGD(model.parameters(), lr=lr)
    # return torch.optim.Adam(model.parameters(),
    #                         lr=lr,
    #                         betas=tuple(betas),
    #                         eps=eps)


def build_scheduler(optimizer, **kwargs):
    return torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def build_criterion(**kwargs):
    # return nn.NLLLoss()
    return nn.CrossEntropyLoss()


def build_model(device, N, d_model, d_ff, h, dropout, src_vocab, tgt_vocab,
                **kwargs):
    def to_parallel(model, device):
        return nn.DataParallel(model) if (device.type == "cuda") else model

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
    return to_parallel(model, device)


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
                  **kwargs)

            val_loss = evaluate(model=model,
                                criterion=criterion,
                                data_source=val_data,
                                **kwargs)

            log('-' * 89)
            log(f'| end of epoch {epoch:3d} '
                f'| time: {(time.time() - epoch_start_time):5.2f}s '
                f'| valid loss {val_loss:5.2f} '
                f'| valid ppl {math.exp(val_loss):8.2f}')
            log('-' * 89)

            # Save the model if the validation loss is the best we've seen so
            # far.
            if val_loss < best_val_loss:
                with open(save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss

            scheduler.step()
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
    log(f'| End of training | test loss {test_loss:5.2f} '
        f'| test ppl {math.exp(test_loss):8.2f}')
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


def train(epoch, model, data_source, criterion, optimizer, scheduler,
          log_interval, **kwargs):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    batches = get_batches(data=data_source, train=True, **kwargs)

    for i, batch in enumerate(batches):
        data, targets = batch.src, batch.tgt
        optimizer.zero_grad()
        output = model.forward(data, targets)
        output = output.view(-1, output.size(-1))
        targets = targets.view(-1)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            log(f'| epoch {epoch:3d} | {i:5d}/{len(batches):5d} batches '
                f'| lr {scheduler.get_last_lr()[0]:02.2f} '
                f'| ms/batch {elapsed * 1000 / log_interval:5.2f} '
                f'| loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
            total_loss = 0
            start_time = time.time()
