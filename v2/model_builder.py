import torch
import torch.nn as nn
from commons.log import log

from .dataset import build_dataset


class ModelBuilder():
    """
    This code was based on the links:
    - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    - https://github.com/pytorch/examples/tree/master/word_language_model

    Other links:
    - http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self):
        pass

    def build(self, cuda, dataset_args, model_args, training_args, **kwargs):
        # ------------------------------------------------
        # Prepare device:
        # ------------------------------------------------
        device = self.prepare_device(cuda)

        # ------------------------------------------------
        # Load data:
        # ------------------------------------------------
        train_data, val_data, test_data = build_dataset(**dataset_args)
        src_vocab, tgt_vocab, file_vocab = self.get_vocabs(train_data)

        # ------------------------------------------------
        # Build model:
        # ------------------------------------------------
        model = self.build_model(device=device,
                                 src_vocab=src_vocab,
                                 tgt_vocab=tgt_vocab,
                                 **model_args)

        criterion = self.build_criterion()
        optimizer = self.build_optimizer(model=model, **training_args)
        scheduler = self.build_scheduler(optimizer=optimizer, **training_args)

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
            "tgt_vocab": tgt_vocab,
            "file_vocab": file_vocab
        }

    def prepare_device(self, cuda):
        if torch.cuda.is_available():
            if not cuda:
                log("WARNING: You have a CUDA device, so you should probably "
                    "run with --cuda")

        return torch.device("cuda" if cuda else "cpu")

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

    def build_model(self, device, src_vocab, tgt_vocab, model_type, **kwargs):
        def to_parallel(model, device):
            return nn.DataParallel(model) if (device.type == "cuda") else model

        MODEL_BUILDER = {
            "Transformer": self.create_transformer,
            "LSTM": self.create_rnn,
            "GRU": self.create_rnn,
            "RNN_TANH": self.create_rnn,
            "RNN_RELU": self.create_rnn
        }

        if model_type in MODEL_BUILDER:
            builder = MODEL_BUILDER[model_type]
            model = builder(model_type=model_type,
                            src_vocab=src_vocab,
                            tgt_vocab=tgt_vocab,
                            **kwargs).to(device)
        else:
            raise Exception(f"Unsupported model type: '{model_type}'")

        return to_parallel(model, device)

    def create_transformer(self, N, d_model, d_ff, h, dropout, src_vocab,
                           tgt_vocab, **kwargs):
        from .model import TransformerModel
        return TransformerModel(d_model=d_model,
                                nhead=h,
                                num_encoder_layers=N,
                                num_decoder_layers=N,
                                dim_feedforward=d_ff,
                                dropout=dropout,
                                src_ntoken=len(src_vocab),
                                tgt_ntoken=len(tgt_vocab))

    def create_rnn(self, model_type, N, d_model, d_ff, dropout, src_vocab,
                   tgt_vocab, **kwargs):
        from .model import RNNModel
        args = {
            "rnn_type": model_type,
            "src_ntoken": len(src_vocab),
            "tgt_ntoken": len(tgt_vocab),
            "ninp": d_model,
            "nhid": d_ff,
            "nlayers": N,
            "dropout": dropout,
            "tie_weights": False
        }  # FIXME: adjust this mapping
        return RNNModel(**args)
