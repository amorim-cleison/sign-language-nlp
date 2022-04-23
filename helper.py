import random
from pydoc import locate

import numpy as np
import torch
from commons.log import log
from commons.util import normpath, save_args, create_if_missing
from sklearn.model_selection import *
from skorch.callbacks import (Checkpoint, EarlyStopping, EpochScoring,
                              GradientNormClipping, LoadInitState, LRScheduler)
from skorch.dataset import CVSplit

from dataset import AslDataset
import model.util as util


def setup_seed(seed, **kwargs):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def prepare_device(cuda):
    if torch.cuda.is_available():
        if not cuda:
            log("WARNING: You have a CUDA device, so you should probably "
                "run with --cuda")

    return torch.device("cuda" if cuda else "cpu")


def dump_args(args):
    _dir = args['workdir']
    create_if_missing(_dir)
    save_args(args, normpath(f"{_dir}/config.yaml"))


def build_net_params(training_args, model_args, model, optimizer, criterion,
                     mode, callbacks, callbacks_names, device, dataset,
                     optimizer_args, criterion_args, **kwargs):
    src_vocab = dataset.vocab_X
    tgt_vocab = dataset.vocab_y

    # Train-split:
    assert ("valid_size"
            in training_args), "`valid_size` is a required parameter"
    train_split = CVSplit(training_args["valid_size"])

    # Callbacks args:
    _callbacks_args = build_callbacks_args(model=model,
                                           mode=mode,
                                           callbacks_names=callbacks_names,
                                           **training_args,
                                           **kwargs)

    # Module args:
    _module_args = prefix_args("module",
                               ensure_list=False,
                               batch_first=dataset.batch_first,
                               src_vocab=src_vocab,
                               tgt_vocab=tgt_vocab,
                               device=device,
                               **model_args)

    # Module args:
    _optimizer_args = prefix_args("optimizer",
                                  ensure_list=False,
                                  **optimizer_args)

    # Criterion args:
    criterion_args["ignore_index"] = util.get_pad_idx(tgt_vocab)
    _criterion_args = prefix_args("criterion",
                                  ensure_list=False,
                                  **criterion_args)

    # Iterators args:
    iterators_args = {
        "collate_fn": collate_data,
        # "num_workers": 4,
        # "shuffle": True
    }
    _iterator_train_args = prefix_args("iterator_train",
                                       ensure_list=False,
                                       **iterators_args)
    _iterator_valid_args = prefix_args("iterator_valid",
                                       ensure_list=False,
                                       **iterators_args)
    # Other args:
    KEYS_FOR_NET = [
        "lr", "max_epochs", "batch_size", "predict_nonlinearity", "warm_start",
        "verbose"
    ]
    _net_args = filter_by_keys(training_args, keys_to_filter=KEYS_FOR_NET)

    return {
        "device": device,
        "module": locate(model),
        "optimizer": locate(optimizer),
        "criterion": locate(criterion),
        "callbacks": callbacks,
        "train_split": train_split,
        "dataset": AslDataset,
        **_net_args,
        **_module_args,
        **_optimizer_args,
        **_criterion_args,
        **_callbacks_args,
        **_iterator_train_args,
        **_iterator_valid_args
    }


def build_grid_params(grid_args, callbacks_names, model, workdir, scoring,
                      verbose, n_jobs, **kwargs):
    def unpack(callbacks_names,
               model,
               workdir,
               scoring,
               verbose,
               n_jobs,
               cv,
               training_args={},
               model_args={},
               optimizer_args={},
               criterion_args={},
               **kwargs):
        # Callbacks args:
        _callbacks_args = build_callbacks_args(model=model,
                                               mode="grid",
                                               workdir=workdir,
                                               callbacks_names=callbacks_names,
                                               ensure_list=True,
                                               **training_args)

        # Module args:
        _module_args = prefix_args("module", ensure_list=True, **model_args)

        # Optimizer args:
        _optimizer_args = prefix_args("optimizer",
                                      ensure_list=True,
                                      **optimizer_args)

        # Criterion args:
        _criterion_args = prefix_args("criterion",
                                      ensure_list=True,
                                      **criterion_args)

        # Training args:
        _training_args = prefix_args(None, ensure_list=True, **training_args)

        # Other args:
        KEYS_FOR_GRID = [
            "n_jobs", "refit", "verbose", "pre_dispatch", "return_train_score"
        ]
        _grid_args = filter_by_keys(kwargs, keys_to_filter=KEYS_FOR_GRID)

        return {
            "refit": True,
            "cv": cv,
            "verbose": verbose,
            "scoring": scoring,
            "n_jobs": n_jobs,
            "error_score": "raise",
            **_grid_args, "param_grid": {
                **_module_args,
                **_optimizer_args,
                **_criterion_args,
                **_callbacks_args,
                **_training_args
            }
        }

    return unpack(callbacks_names=callbacks_names,
                  model=model,
                  workdir=workdir,
                  scoring=scoring,
                  verbose=verbose,
                  n_jobs=n_jobs,
                  **grid_args)


def build_callbacks(mode,
                    workdir,
                    resumable,
                    scoring,
                    early_stopping=None,
                    gradient_clipping=None,
                    lr_scheduler=None,
                    **kwargs):
    has_valid = True
    monitor = "valid" if has_valid else "train"

    # Callbacks:
    callbacks = []

    # Checkpoint saving:
    checkpoint = Checkpoint(monitor=f"{monitor}_loss_best", dirname=workdir)
    callbacks.append(("checkpoint", checkpoint))

    # Init state (resume):
    if resumable and mode == "train":
        callbacks.append(("resume_state", LoadInitState(checkpoint)))

    # Early stopping:
    if early_stopping:
        callbacks.append(("early_stopping",
                          EarlyStopping(**early_stopping,
                                        monitor=f"{monitor}_loss",
                                        lower_is_better=True,
                                        sink=log)))

    # Gradient clip:
    if gradient_clipping:
        callbacks.append(
            ("gradient_clipping", GradientNormClipping(**gradient_clipping)))

    # LR Scoring:
    def lr_score(net, X, y=None):
        return net.optimizer_.param_groups[0]["lr"]

    callbacks.append(("lr_scoring",
                      EpochScoring(scoring=lr_score,
                                   name='lr',
                                   on_train=not has_valid)))

    # LR Scheduler:
    if lr_scheduler:
        callbacks.append(("lr_scheduler",
                          LRScheduler(monitor=f"{monitor}_loss",
                                      step_every="epoch",
                                      **lr_scheduler)))

    # Scoring metric (dynamic):
    if not isinstance(scoring, list):
        scoring = [scoring]

    for score in scoring:
        wrapper = ScoringWrapper(score)

        callbacks.append(
            (f"score_{score}",
             EpochScoring(scoring=wrapper,
                          name=f"{monitor}_{score}",
                          on_train=not has_valid,
                          lower_is_better=not wrapper.greater_is_better)))

    # Callbacks names:
    callbacks_names = [c[0] for c in callbacks]

    return callbacks, callbacks_names


def build_callbacks_args(callbacks_names, ensure_list=False, **kwargs):
    __standard_callbacks = ["print_log"]
    callbacks_args = filter_by_keys(kwargs,
                                    callbacks_names + __standard_callbacks)
    return prefix_args("callbacks", ensure_list=ensure_list, **callbacks_args)


def collate_data(data):
    X, y = zip(*data)

    if len(X[0]) == 3:
        X, X_lengths, _ = zip(*X)
    elif len(X[0]) == 2:
        X, X_lengths = zip(*X)

    X = torch.tensor(X, dtype=torch.long)
    X_lengths = torch.tensor(X_lengths, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    return {"X": X, "lengths": X_lengths, "y": y}, y


def format_dir(dir, **kwargs):
    return normpath(dir.format(**kwargs)) if dir is not None else ''


def filter_by_keys(map, keys_to_filter, not_in=False):
    return dict(
        filter(lambda o: not (o[0] in keys_to_filter) == not_in, map.items()))


def is_cv_for_net(cross_validator):
    return isinstance(cross_validator, (int, float, CVSplit))


def prefix_args(prefix, ensure_list=False, output=None, **kwargs):
    if output is None:
        output = {}

    for k, v in kwargs.items():
        name = k if prefix is None else f"{prefix}__{k}"

        if isinstance(v, dict):
            prefix_args(prefix=name,
                        output=output,
                        ensure_list=ensure_list,
                        **v)
        else:
            if ensure_list and not isinstance(v, list):
                v = [v]
            output[name] = v
    return output


def balance_dataset(dataset, seed):
    import math
    from collections import Counter
    from statistics import mean

    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline
    from imblearn.under_sampling import RandomUnderSampler

    from dataset import AslDataset

    def compute_sampling(data, mode="under"):
        def smooth_v(v, u, sign):
            tmp = round(u + math.log(v))
            return v if (v * sign) > (tmp * sign) else tmp

        _signs = {"under": -1, "over": +1}
        assert (mode in _signs), "Invalid mode"

        u = mean(data.values())
        sign = _signs[mode]
        return {k: smooth_v(v, u, sign) for (k, v) in data.items()}

    # Original data:
    X, y = dataset.X().to_array(), dataset.y().to_array()
    original = Counter(y)

    # Compute samplings:
    under_sampling = compute_sampling(original, "under")
    over_sampling = compute_sampling(under_sampling, "over")

    # Prepare pipeline:
    rus = RandomUnderSampler(sampling_strategy=under_sampling,
                             random_state=seed,
                             replacement=False)
    ros = RandomOverSampler(sampling_strategy=over_sampling, random_state=seed)
    pipeline = Pipeline(steps=[(type(rus).__name__, rus),
                               (type(ros).__name__, ros)],
                        verbose=True)

    # Resample data:
    X_res, y_res = pipeline.fit_resample(X, y)
    dataset_res = AslDataset(dataset=dataset, X=X_res, y=y_res)

    return dataset_res


class ScoringWrapper:
    def __init__(self, score_func):
        from sklearn.metrics import get_scorer
        self._score_func = score_func
        self.scorer = get_scorer(score_func)
        # FIXME: add support to externalize scoring kwargs/options:
        self.scorer._kwargs["zero_division"] = 0

    def __call__(self, estimator, X, y_true, sample_weight=None):
        return self.scorer(estimator, X, y_true, sample_weight)

    def __repr__(self):
        return f"{type(self).__name__}('{self._score_func}')"

    @property
    def greater_is_better(self):
        return (self.scorer._sign == 1)
