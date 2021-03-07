import math
import time
import torch
import torch.nn as nn
from commons.log import log
from commons.util import normpath
from .dataset import build_dataset, build_iterator, PAD_WORD
from .util import log_step, save_step


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
                  checkpoint_dir=checkpoint_dir,
                  **kwargs)

            val_loss, val_acc = evaluate(model=model,
                                         criterion=criterion,
                                         data_source=val_data,
                                         **kwargs)

            step_data = {
                "epoch": f"{epoch:3d}",
                "time": f"{(time.time() - epoch_start_time):5.2f}s",
                "acc": f"{val_acc:5.2f}",
                "loss": f"{val_loss:5.2f}",
                "ppl": f"{math.exp(val_loss):8.2f}"
            }
            log_step("valid",  sep="-", **step_data)
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


def run_test(criterion, test_data, checkpoint_dir, **kwargs):
    save = normpath(f"{checkpoint_dir}/weights.pt")

    # Load the best saved model.
    with open(save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss, test_acc = evaluate(model=model,
                                   criterion=criterion,
                                   data_source=test_data,
                                   **kwargs)
    step_data = {
        "acc": f"{test_acc:5.2f}",
        "loss": f"{test_loss:5.2f}",
        "ppl": f"{math.exp(test_loss):8.2f}"
    }
    log_step("test", sep="=", **step_data)
    save_step("test", checkpoint_dir, **step_data)


def evaluate(model, criterion, data_source, **kwargs):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total_correct = 0
    batches = get_batches(data=data_source, train=False, **kwargs)

    with torch.no_grad():
        for i, batch in enumerate(batches):
            data, targets = batch.src, batch.tgt
            output = model.forward(data, targets)
            output = output.view(-1, output.size(-1))
            targets = targets.view(-1)

            # Loss:
            # TODO: review this:
            # total_loss += len(data) * criterion(output, targets).item()
            total_loss += data.size(-1) * criterion(output, targets).item()

            # Accuracy:
            total_correct += (torch.argmax(output, dim=-1) == targets).sum()

    loss = total_loss / (len(data_source) - 1)
    accuracy = total_correct / (len(data_source) - 1)
    return loss, accuracy


def train(epoch, model, data_source, criterion, optimizer, scheduler,
          log_interval, checkpoint_dir, **kwargs):
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
            step_data = {
                "epoch": f"{epoch:3d}",
                "batch": f"{i:5d} /{len(batches):5d}",
                "lr": f"{scheduler.get_last_lr()[0]:02.5f}",
                "ms/batch": f"{elapsed * 1000 / log_interval:5.2f}",
                "loss": f"{cur_loss:5.2f}",
                "ppl": f"{math.exp(cur_loss):8.2f}"
            }
            log_step("train", **step_data)
            save_step("train", checkpoint_dir, **step_data)
            total_loss = 0
            start_time = time.time()
