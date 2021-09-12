import math
import time

import torch
from commons.log import log
from commons.util import normpath

from dataset import CustomIterator
from util import (log_data, log_model, log_step, save_eval_outputs, save_step)


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
        try:
            self.do_run(debug=debug, **training_args)
        except Exception as e:
            raise Exception(f"Failed to run model: {repr(e)}")

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

    def repackage_hidden(self, h):
        """
        Wraps hidden states in new Tensors, to detach them from their history.
        """
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

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
            if best_model.model_type in [
                    'RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU'
            ]:
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

        if self.model.model_type != 'Transformer':
            hidden = self.model.init_hidden(batch_size)
        else:
            hidden = None

        for i, batch in enumerate(batches):
            # Data:
            src, tgt = batch.src, batch.tgt

            # Forward:
            self.optimizer.zero_grad()
            output = self.forward(batch=batch,
                                  input=src,
                                  targets=tgt,
                                  hidden=hidden)

            # Loss and optimization:
            loss = self.compute_loss(output=output, targets=tgt)
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

        if model.model_type != 'Transformer':
            hidden = model.init_hidden(batch_size)
        else:
            hidden = None

        with torch.no_grad():
            for i, batch in enumerate(batches):
                # Data:
                src, tgt, files = batch.src, batch.tgt, batch.file

                # Forward:
                output = self.forward(batch=batch,
                                      input=src,
                                      targets=tgt,
                                      hidden=hidden)

                # Loss:
                total_loss += self.compute_loss(output=output,
                                                targets=tgt).item()

                # Accuracy:
                total_acc += self.compute_accuracy(output=output,
                                                   targets=tgt).item()

                # Save outputs:
                output = output[-1]
                tgt = tgt[-1]
                files = files[-1]
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

    def forward(self, batch, input, targets, hidden):
        if self.model.model_type == 'Transformer':
            src_mask = generate_mask(input).to(self.device)
            tgt_mask = generate_mask(targets).to(self.device)
            src_padding_mask = \
                generate_padding_mask(input, self.src_vocab).to(self.device)
            tgt_padding_mask = \
                generate_padding_mask(targets, self.tgt_vocab).to(self.device)

            output = self.model.forward(src=input,
                                        tgt=targets,
                                        src_mask=src_mask,
                                        tgt_mask=tgt_mask,
                                        src_key_padding_mask=src_padding_mask,
                                        tgt_key_padding_mask=tgt_padding_mask)
        else:
            output, hidden = self.model.forward(input=input, hidden=hidden)
            hidden = self.repackage_hidden(hidden)
        return output

    def compute_loss(self, output, targets):
        # total_loss += len(data) * criterion(output, targets).item()
        # output = output.view(-1, output.size(-1))
        # targets = targets.view(-1)
        output = output[-1]
        targets = targets[-1]
        return self.criterion(output, targets)

    def compute_accuracy(self, output, targets):
        # As the targets are shifted right (the first index is BOS token),
        # we will consider since the second index:
        output = output[1:]
        targets = targets[1:]

        output_labels = torch.argmax(output, dim=-1)
        corrects = (output_labels == targets)
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
