import torch
import torch.nn as nn
from commons.log import log

from model import GRU, LSTM, RNNRelu, RNNTanh, Transformer
from model.util import get_pad_idx


class ModelBuilder():
    MODELS = {
        "transformer": Transformer,
        "gru": GRU,
        "lstm": LSTM,
        "rnn_relu": RNNRelu,
        "rnn_tanh": RNNTanh
    }
    """
    This code was based on the links:
    - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    - https://github.com/pytorch/examples/tree/master/word_language_model

    Other links:
    - http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, src_vocab, tgt_vocab, **kwargs):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def build(self, **kwargs):
        return self.do_build(**kwargs)

    def do_build(self, cuda, model_args, training_args, **kwargs):
        # ------------------------------------------------
        # Prepare device:
        # ------------------------------------------------
        device = self.prepare_device(cuda)

        # ------------------------------------------------
        # Build model:
        # ------------------------------------------------
        model = self.build_model(device=device,
                                 src_vocab=self.src_vocab,
                                 tgt_vocab=self.tgt_vocab,
                                 **model_args)

        criterion = self.build_criterion()
        optimizer = self.build_optimizer(model=model, **training_args)
        scheduler = self.build_scheduler(optimizer=optimizer, **training_args)

        return {
            "device": device,
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "scheduler": scheduler
        }

    def prepare_device(self, cuda):
        if torch.cuda.is_available():
            if not cuda:
                log("WARNING: You have a CUDA device, so you should probably "
                    "run with --cuda")

        return torch.device("cuda" if cuda else "cpu")

    def build_criterion(self, **kwargs):
        # return nn.NLLLoss()
        pad_idx = get_pad_idx(self.tgt_vocab)
        return nn.CrossEntropyLoss(ignore_index=pad_idx)

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
                                               gamma=lr_step_gamma,
                                               verbose=True)

    def build_model(self, device, src_vocab, tgt_vocab, model_type, **kwargs):
        def to_parallel(model, device):
            return nn.DataParallel(model) if (device.type == "cuda") else model

        assert (model_type.lower() in self.MODELS), \
            f"Unsupported model type: '{model_type}'"

        extra_args = {
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
            "device": device
        }
        _cls = self.MODELS[model_type.lower()]
        model = _cls(**extra_args, **kwargs).to(device)
        return to_parallel(model, device)
