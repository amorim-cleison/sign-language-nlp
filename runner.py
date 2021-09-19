import math
import time

import torch
from commons.log import log
from commons.util import normpath

from dataset import CustomIterator
from util import (log_data, log_model, log_step, save_eval_outputs, save_step)
from model.util import get_pad_idx, pad_to_shape


class Runner():
    """
    This code was based on the links:
    - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    - https://github.com/pytorch/examples/tree/master/word_language_model

    Other links:
    - http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, device, model, criterion, optimizer, scheduler,
                 train_data, val_data, test_data, src_vocab, tgt_vocab,
                 file_vocab, **kwargs):
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
        self.file_vocab = file_vocab

    def run(self, debug, training_args, **kwargs):
        # try:
        self.do_run(debug=debug, **training_args)
        # except Exception as e:
        #     raise Exception(f"Failed to run model: {repr(e)}")

    def do_run(self, seed, debug, epochs, log_interval, checkpoint_dir,
               batch_size, **kwargs):
        self.log_objects()

        # ------------------------------------------------
        # Set random seed manually (for reproducibility):
        # ------------------------------------------------
        self.setup_seed(seed)

        # ------------------------------------------------
        # Train model:
        # ------------------------------------------------
        self.run_training(debug, epochs, log_interval, checkpoint_dir,
                          batch_size)

        # ------------------------------------------------
        # Test model:
        # ------------------------------------------------
        self.run_test(debug, checkpoint_dir, batch_size)

    def setup_seed(self, seed):
        torch.manual_seed(seed)

    def get_batches(self, data, device, batch_size, train):
        return CustomIterator(data,
                              batch_size=batch_size,
                              device=device,
                              repeat=False,
                              sort_key=lambda x: (len(x.src), len(x.tgt)),
                              train=train)

    # def repackage_hidden(self, h):
    #     """
    #     Wraps hidden states in new Tensors, to detach them from their history.
    #     """
    #     if isinstance(h, torch.Tensor):
    #         return h.detach()
    #     else:
    #         return tuple(self.repackage_hidden(v) for v in h)

    def run_training(self, debug, epochs, log_interval, checkpoint_dir,
                     batch_size):
        best_val_loss = float("inf")
        checkpoint_path = normpath(f"{checkpoint_dir}/weights.pt")

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()
                self.train(debug=debug,
                           epoch=epoch,
                           data_source=self.train_data,
                           log_interval=log_interval,
                           checkpoint_dir=checkpoint_dir,
                           batch_size=batch_size)

                val_loss, val_acc, val_ppl = self.evaluate(
                    debug=debug,
                    epoch=epoch,
                    model=self.model,
                    data_source=self.val_data,
                    checkpoint_dir=checkpoint_dir,
                    batch_size=batch_size)

                step_data = {
                    "epoch": f"{epoch:3d}",
                    "time": f"{(time.time() - epoch_start_time):5.2f}s",
                    "acc": f"{val_acc:5.2f}",
                    "loss": f"{val_loss:5.2f}",
                    "ppl": f"{val_ppl:8.2f}"
                }
                log_step("valid", sep="-", **step_data)
                save_step("valid", checkpoint_dir, **step_data)

                # Save the model if the validation loss is the best we've seen
                # so far.
                if val_loss < best_val_loss:
                    with open(checkpoint_path, 'wb') as f:
                        torch.save(self.model, f)
                    best_val_loss = val_loss
                else:
                    self.scheduler.step()

                # self.scheduler.step()
        except KeyboardInterrupt:
            log('-' * 100)
            log('Exiting from training early')

    def run_test(self, debug, checkpoint_dir, batch_size):
        save = normpath(f"{checkpoint_dir}/weights.pt")

        # Load the best saved model.
        with open(save, 'rb') as f:
            best_model = torch.load(f)

            # after load the rnn params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward
            # pass. Currently, only rnn model supports flatten_parameters
            # function.
            if self.is_rnn(best_model):
                best_model.rnn.flatten_parameters()

        # Run on test data.
        test_loss, test_acc, test_ppl = self.evaluate(
            debug=debug,
            model=best_model,
            data_source=self.test_data,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size)
        step_data = {
            "acc": f"{test_acc:5.2f}",
            "loss": f"{test_loss:5.2f}",
            "ppl": f"{test_ppl:8.2f}"
        }
        log_step("test", sep="=", **step_data)
        save_step("test", checkpoint_dir, **step_data)

    def train(self, debug, epoch, data_source, log_interval, checkpoint_dir,
              batch_size):
        # Turn on training mode which enables dropout.
        self.model.train()
        total_loss = 0.
        start_time = time.time()
        batches = self.get_batches(data=data_source,
                                   train=True,
                                   device=self.device,
                                   batch_size=batch_size)

        for i, batch in enumerate(batches):
            self.optimizer.zero_grad()

            # Forward:
            output = self.forward(model=self.model, batch=batch)

            # Loss and optimization:
            loss = self.compute_loss(output=output, targets=batch.tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                step_data = {
                    "epoch": f"{epoch:3d}",
                    "batch": f"{i:5d} /{len(batches):5d}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:02.5f}",
                    "ms/batch": f"{elapsed * 1000 / log_interval:5.2f}",
                    "loss": f"{cur_loss:5.2f}",
                    "ppl": f"{math.exp(cur_loss):8.2f}"
                }
                log_step("train", **step_data)
                save_step("train", checkpoint_dir, **step_data)
                total_loss = 0
                start_time = time.time()

                # Early stop when debugging:
                if debug:
                    break

    def evaluate(self,
                 debug,
                 model,
                 data_source,
                 checkpoint_dir,
                 batch_size,
                 epoch=None):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        total_acc = 0.
        batches = self.get_batches(data=data_source,
                                   train=False,
                                   device=self.device,
                                   batch_size=batch_size)

        with torch.no_grad():
            for i, batch in enumerate(batches):
                # Forward:
                output = self.forward(model=model, batch=batch)

                # Loss:
                total_loss += self.compute_loss(output=output,
                                                targets=batch.tgt).item()

                # Accuracy:
                total_acc += self.compute_accuracy(output=output,
                                                   targets=batch.tgt).item()

                # Save outputs:
                output = output[-1]
                tgt = batch.tgt[-1]
                files = batch.file[-1]
                save_eval_outputs(outputs=output,
                                  targets=tgt,
                                  files=files,
                                  tgt_vocab=self.tgt_vocab,
                                  file_vocab=self.file_vocab,
                                  dir=checkpoint_dir,
                                  epoch=epoch)

                # Early stop when debugging:
                if debug:
                    break

        loss = total_loss / len(batches)
        accuracy = total_acc / len(batches)
        ppl = math.exp(loss)
        return loss, accuracy, ppl

    def forward(self, model, batch):
        if self.is_rnn(model):
            # hidden = self.repackage_hidden(hidden)
            hidden = model.init_hidden(batch_size=batch.batch_size)
            output, _ = model.forward(input=batch.src, hidden=hidden)
        else:
            output = model.forward(input=batch.src, targets=batch.tgt)
        return output

    def compute_loss(self, output, targets):
        targets = pad_to_shape(targets, output.shape, self.tgt_vocab)
        targets = targets.view(-1)
        output = output.view(-1, output.size(-1))
        return self.criterion(output, targets)

    def compute_accuracy(self, output, targets):
        # As the targets are shifted right (the first index is BOS token),
        # we will consider since the second index:
        targets = pad_to_shape(targets, output.shape, self.tgt_vocab)
        targets = targets.view(-1)
        output = output.view(-1, output.size(-1))
        output = output.argmax(-1)

        corrects = (output == targets)
        total = targets.nelement()
        return corrects.sum() / total

    def log_objects(self):
        # ------------------------------------------------
        # Log objects created:
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

    def is_rnn(self, model):
        from model.base import RNNModel
        return isinstance(model, RNNModel)
