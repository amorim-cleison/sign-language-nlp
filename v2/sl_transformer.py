import math
import time
import torch
import torch.nn as nn
from commons.log import log
from commons.util import normpath
from .dataset import build_dataset, build_iterator, PAD_WORD
from .util import log_step, save_step
"""
This code was base on the link:
- https://pytorch.org/tutorials/beginner/transformer_tutorial.html

Other links:
- https://github.com/pytorch/examples/tree/master/word_language_model
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""


def run(seed, cuda, config, dataset_args, model_args, training_args,
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
    src_vocab, tgt_vocab = get_vocabs(train_data)

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


def get_vocabs(dataset):
    return dataset.fields["src"].vocab, dataset.fields["tgt"].vocab


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
                        src_ntoken=len(src_vocab),
                        tgt_ntoken=len(tgt_vocab)).to(device)
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
            log_step("valid", sep="-", **step_data)
            # save_step("valid", checkpoint_dir, **step_data)

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


def evaluate(model, criterion, data_source, device, **kwargs):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total_acc = 0
    src_vocab, tgt_vocab = get_vocabs(data_source)
    batches = get_batches(data=data_source,
                          train=False,
                          device=device,
                          **kwargs)

    with torch.no_grad():
        for i, batch in enumerate(batches):
            # Data:
            src, tgt = batch.src, batch.tgt

            # Masks:
            src_mask = None
            tgt_mask = generate_mask(tgt, model).to(device)
            src_padding_mask = generate_padding_mask(src, src_vocab).to(device)
            tgt_padding_mask = generate_padding_mask(tgt, tgt_vocab).to(device)

            # Forward:
            output = model.forward(src=src,
                                   tgt=tgt,
                                   src_mask=src_mask,
                                   tgt_mask=tgt_mask,
                                   src_key_padding_mask=src_padding_mask,
                                   tgt_key_padding_mask=tgt_padding_mask)
            output = output.view(-1, output.size(-1))
            tgt = tgt.view(-1)

            # Loss:
            # TODO: review this. This was done once the loss was nn.NLLLoss()
            # total_loss += len(data) * criterion(output, targets).item()
            total_loss += criterion(output, tgt).item()

            # Accuracy:
            # FIXME: analyze this. calculation considers BOS token
            total_acc += calc_accuracy(output, tgt)

    loss = total_loss / len(batches)
    accuracy = total_acc / len(batches)
    return loss, accuracy


def train(epoch, model, data_source, criterion, optimizer, scheduler,
          log_interval, checkpoint_dir, device, **kwargs):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    src_vocab, tgt_vocab = get_vocabs(data_source)
    batches = get_batches(data=data_source,
                          train=True,
                          device=device,
                          **kwargs)

    for i, batch in enumerate(batches):
        # Data:
        src, tgt = batch.src, batch.tgt

        # Masks:
        src_mask = None
        tgt_mask = generate_mask(tgt, model).to(device)
        src_padding_mask = generate_padding_mask(src, src_vocab).to(device)
        tgt_padding_mask = generate_padding_mask(tgt, tgt_vocab).to(device)

        # Forward step:
        optimizer.zero_grad()
        output = model.forward(src=src,
                               tgt=tgt,
                               src_mask=src_mask,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_padding_mask,
                               tgt_key_padding_mask=tgt_padding_mask)
        output = output.view(-1, output.size(-1))
        tgt = tgt.view(-1)

        # Loss and optimization:
        loss = criterion(output, tgt)
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


def calc_accuracy(output, targets):
    return (torch.argmax(output, dim=-1) == targets).sum() / len(targets)


def generate_mask(data, model):
    """
    Mask ensures that position i is allowed to attend the unmasked
    positions. If a ByteTensor is provided, the non-zero positions are
    not allowed to attend while the zero positions will be unchanged.
    If a BoolTensor is provided, positions with ``True`` are not
    allowed to attend while ``False`` values will be unchanged.
    If a FloatTensor is provided, it will be added to the attention
    weight.
    """
    def generate_square_subsequent_mask(self, sz: int):
        r"""
        Generate a square mask for the sequence. The masked positions are
        filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        
        Extracted from `torch.nn.Transformer.generate_square_subsequent_mask`.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    mask = generate_square_subsequent_mask(data.size(0))
    mask = (mask != float(0.0)).bool()
    return mask


def generate_padding_mask(data, vocab):
    """
    Padding mask provides specified elements in the key to be ignored
    by the attention. If a ByteTensor is provided, the non-zero
    positions will be ignored while the zero positions will be
    unchanged. If a BoolTensor is provided, the positions with the
    value of ``True`` will be ignored while the position with the
    value of ``False`` will be unchanged.
    """
    pad_idx = vocab.stoi[PAD_WORD]
    mask = (data == pad_idx).transpose(0, 1).bool()
    return mask
