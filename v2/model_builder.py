import math
import time

import torch
import torch.nn as nn
from commons.log import log
from commons.util import normpath

from .dataset import build_dataset, build_iterator, PAD_WORD
from .util import (generate_mask, generate_padding_mask, log_step, log_model,
                   log_data, save_eval_outputs, save_step)


class ModelBuilder():
    """
    This code was based on the links:
    - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    - https://github.com/pytorch/examples/tree/master/word_language_model

    Other links:
    - http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, debug, model_type, cuda, dataset_args, model_args,
                 training_args, transfer_learning_args, **kwargs):
        self.debug = debug
        self.model_type = model_type
        self.cuda = cuda
        self.dataset_args = dataset_args
        self.model_args = model_args
        self.training_args = training_args

    def build(self):
        # ------------------------------------------------
        # Prepare CUDA:
        # ------------------------------------------------
        device = self.prepare_cuda()

        # ------------------------------------------------
        # Load data:
        # ------------------------------------------------
        train_data, val_data, test_data = build_dataset(device=device,
                                                        **self.dataset_args,
                                                        **self.training_args)
        src_vocab, tgt_vocab, _ = self.get_vocabs(train_data)

        # ------------------------------------------------
        # Build model:
        # ------------------------------------------------
        model = self.build_model(device=device,
                                 model_type=self.model_type,
                                 src_vocab=src_vocab,
                                 tgt_vocab=tgt_vocab,
                                 **self.model_args)

        criterion = self.build_criterion(**self.training_args)
        optimizer = self.build_optimizer(model=model, **self.training_args)
        scheduler = self.build_scheduler(optimizer=optimizer,
                                         **self.training_args)

        return {
            "device": device,
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "train_data": train_data,
            "val_data": val_data,
            "test_data": test_data,
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab
        }

    def prepare_cuda(self):
        if torch.cuda.is_available():
            if not self.cuda:
                log("WARNING: You have a CUDA device, so you should probably "
                    "run with --cuda")

        return torch.device("cuda" if self.cuda else "cpu")

    def get_vocabs(self, dataset):
        return dataset.fields["src"].vocab,\
            dataset.fields["tgt"].vocab,\
            dataset.fields["file"].vocab

    def build_criterion(self, **kwargs):
        # return nn.NLLLoss()
        return nn.CrossEntropyLoss(ignore_index=1)

    def build_optimizer(self, model, lr, **kwargs):
        # return torch.optim.Adam(model.parameters(),
        #                         lr=lr,
        #                         betas=tuple(betas),
        #                         eps=eps)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def build_scheduler(self, optimizer, lr_step_size, lr_step_gamma,
                        **kwargs):
        return torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=lr_step_size,
                                               gamma=lr_step_gamma)

    def build_model(self, device, model_type, N, d_model, d_ff, h, dropout,
                    src_vocab, tgt_vocab, **kwargs):
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
