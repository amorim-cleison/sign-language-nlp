import math
import time

import torch
import torch.nn as nn
from commons.log import log
from commons.util import normpath

from .dataset import build_dataset, build_iterator, PAD_WORD
from .util import (generate_mask, generate_padding_mask, log_step, log_model,
                   log_data, save_eval_outputs, save_step)


class ModelRunner():
    """
    This code was based on the links:
    - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    - https://github.com/pytorch/examples/tree/master/word_language_model

    Other links:
    - http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, debug, seed, training_args, device, model, criterion,
                 optimizer, scheduler, train_data, val_data, test_data,
                 src_vocab, tgt_vocab, **kwargs):
        self.debug = debug
        self.seed = seed
        self.training_args = training_args

        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def run(self):
        # ------------------------------------------------
        # Set random seed manually (for reproducibility):
        # ------------------------------------------------
        torch.manual_seed(self.seed)

        # ------------------------------------------------
        # Set random seed manually (for reproducibility):
        # ------------------------------------------------
        log_data({
            "train": self.train_data,
            "val": self.val_data,
            "test": self.test_data,
            "src vocab": self.src_vocab,
            "tgt vocab": self.tgt_vocab
        })

        log_model({
            "model": self.model,
            "criterion": self.criterion,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler
        })

        # ------------------------------------------------
        # Train model:
        # ------------------------------------------------
        self.run_training(model=self.model,
                          criterion=self.criterion,
                          optimizer=self.optimizer,
                          scheduler=self.scheduler,
                          train_data=self.train_data,
                          val_data=self.val_data,
                          device=self.device,
                          **self.training_args)

        # ------------------------------------------------
        # Test model:
        # ------------------------------------------------
        self.run_test(criterion=self.criterion,
                      test_data=self.test_data,
                      device=self.device,
                      **self.training_args)

    # FIXME: check if its possible remove multiple calls to get_vocab
    def get_vocabs(self, dataset):
        return dataset.fields["src"].vocab,\
            dataset.fields["tgt"].vocab,\
            dataset.fields["file"].vocab

    def get_batches(self, data, device, batch_size, train, **kwargs):
        return build_iterator(data, batch_size, device, train)

    def repackage_hidden(self, h):
        """
        Wraps hidden states in new Tensors, to detach them from their history.
        """
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def run_training(self, model, epochs, criterion, optimizer, scheduler,
                     train_data, val_data, log_interval, checkpoint_dir,
                     **kwargs):
        best_val_loss = float("inf")
        checkpoint_path = normpath(f"{checkpoint_dir}/weights.pt")

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()
                self.train(epoch=epoch,
                           model=model,
                           data_source=train_data,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           log_interval=log_interval,
                           checkpoint_dir=checkpoint_dir,
                           **kwargs)

                val_loss, val_acc = self.evaluate(
                    epoch=epoch,
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

                # Save the model if the validation loss is the best we've seen
                # so far.
                if val_loss < best_val_loss:
                    with open(checkpoint_path, 'wb') as f:
                        torch.save(model, f)
                    best_val_loss = val_loss

                scheduler.step()
        except KeyboardInterrupt:
            log('-' * 100)
            log('Exiting from training early')

    def run_test(self, criterion, test_data, checkpoint_dir,
                 **kwargs):
        save = normpath(f"{checkpoint_dir}/weights.pt")

        # Load the best saved model.
        with open(save, 'rb') as f:
            model = torch.load(f)

            # after load the rnn params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward
            # pass. Currently, only rnn model supports flatten_parameters
            # function.
            if model.model_type in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
                model.rnn.flatten_parameters()

        # Run on test data.
        test_loss, test_acc = self.evaluate(model=model,
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

    def train(self, epoch, model, data_source, criterion,
              optimizer, scheduler, log_interval, checkpoint_dir, device,
              batch_size, **kwargs):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.
        start_time = time.time()
        src_vocab, tgt_vocab, _ = self.get_vocabs(data_source)
        batches = self.get_batches(data=data_source,
                                   train=True,
                                   device=device,
                                   batch_size=batch_size,
                                   **kwargs)

        if model.model_type != 'Transformer':
            hidden = model.init_hidden(batch_size)

        for i, batch in enumerate(batches):
            # Data:
            src, tgt = batch.src, batch.tgt

            # Forward:
            optimizer.zero_grad()

            output = self.forward(model, device, batch, src, tgt,
                                  src_vocab, tgt_vocab)

            # Loss and optimization:
            loss = self.compute_loss(criterion, output, tgt)
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

    def evaluate(self,
                 model,
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
        src_vocab, tgt_vocab, file_vocab = self.get_vocabs(data_source)
        batches = self.get_batches(data=data_source,
                                   train=False,
                                   device=device,
                                   batch_size=batch_size,
                                   **kwargs)

        if model.model_type != 'Transformer':
            hidden = model.init_hidden(batch_size)

        with torch.no_grad():
            for i, batch in enumerate(batches):
                # Data:
                src, tgt, files = batch.src, batch.tgt, batch.file

                # Forward:
                output = self.forward(model, device, batch, src,
                                      tgt, src_vocab, tgt_vocab)
                # Loss:
                total_loss += self.compute_loss(criterion, output, tgt).item()

                # Accuracy:
                total_acc += self.compute_accuracy(output, tgt).item()

                # Save outputs:
                output = output[-1]
                tgt = tgt[-1]
                files = files[-1]
                save_eval_outputs(output, tgt, files, tgt_vocab, file_vocab,
                                  checkpoint_dir, epoch)

        loss = total_loss / len(batches)
        accuracy = total_acc / len(batches)
        return loss, accuracy

    def forward(self, model, device, batch, src, tgt, src_vocab,
                tgt_vocab):
        if model.model_type == 'Transformer':
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
        return output

    def compute_loss(self, criterion, output, targets):
        # total_loss += len(data) * criterion(output, targets).item()
        # output = output.view(-1, output.size(-1))
        # targets = targets.view(-1)
        output = output[-1]
        targets = targets[-1]
        return criterion(output, targets)

    def compute_accuracy(self, output, targets):
        # As the targets are shifted right (the first index is BOS token),
        # we will consider since the second index:
        output = output[1:]
        targets = targets[1:]

        output_labels = torch.argmax(output, dim=-1)
        corrects = (output_labels == targets)
        total = targets.nelement()
        return corrects.sum() / total
