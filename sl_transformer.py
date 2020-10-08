# import numpy as np
import torch
from torch.autograd import Variable
from model import (LabelSmoothing, make_model, run_epoch)


class SimpleLossCompute:
    # FIXME: implement this
    pass


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # FIXME: implement this
    pass


def run(input_dir, **kwargs):
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(
        model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(
            model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    data = load_data(input_dir)

    for epoch in range(10):
        model.train()
        run_epoch(
            data, model,
            SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(
            run_epoch(
                data, model,
                SimpleLossCompute(model.generator, criterion, None)))

    model.eval()
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))


# TODO: check if can promote this to 'model' file
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size**
                              (-0.5) * min(step**
                                           (-0.5), step * self.warmup**(-1.5)))


def load_data(input_dir):
    import json
    import util as u

    all_data = list()
    files = u.filter_files(input_dir, ext="json")
    # total = len(files)

    for idx, path in enumerate(files):
        # log(f" [{idx + 1} / {total}] Reading '{path}'...", 2)

        with open(path) as file:
            content = json.load(file)
            all_data.append((content, path))
    return all_data
