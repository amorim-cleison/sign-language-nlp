import random
from pydoc import locate

import numpy as np
import torch
from commons.log import log
from commons.util import normpath
from sklearn.model_selection import *
from skorch.callbacks import (Checkpoint, EarlyStopping, EpochScoring,
                              GradientNormClipping, LoadInitState, LRScheduler)
from skorch.dataset import CVSplit

from dataset import AslSliceDataset


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


def build_net_params(training_args, model_args, model, optimizer, criterion,
                     mode, resumable, callbacks, callbacks_names, device,
                     dataset, optimizer_args, criterion_args, cross_validator,
                     **kwargs):
    # Configure params:
    train_split = cross_validator if is_cv_for_net(cross_validator) else None

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
                               src_vocab=dataset.vocab_X,
                               tgt_vocab=dataset.vocab_y,
                               **model_args)

    # Module args:
    _optimizer_args = prefix_args("optimizer",
                                  ensure_list=False,
                                  **optimizer_args)

    # Criterion args:
    _criterion_args = prefix_args("criterion",
                                  ensure_list=False,
                                  **criterion_args)

    # Iterators args:
    iterators_args = {
        "collate_fn": dataset.collate,
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
        **_net_args,
        **_module_args,
        **_optimizer_args,
        **_criterion_args,
        **_callbacks_args,
        **_iterator_train_args,
        **_iterator_valid_args
    }


def build_grid_params(grid_args, callbacks_names, model, workdir, scoring,
                      cross_validator, verbose, **kwargs):
    def unpack(training_args,
               callbacks_names,
               model,
               workdir,
               cross_validator,
               scoring,
               verbose,
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

        # Other args:
        KEYS_FOR_GRID = [
            "n_jobs", "refit", "verbose", "pre_dispatch", "return_train_score"
        ]
        _grid_args = filter_by_keys(kwargs, keys_to_filter=KEYS_FOR_GRID)

        # Scoring:
        _scoring = ScoringWrapper(scoring)

        return {
            "refit": True,
            "cv": cross_validator,
            "verbose": verbose,
            "scoring": _scoring,
            "error_score": "raise",
            **_grid_args, "param_grid": {
                **_module_args,
                **_optimizer_args,
                **_criterion_args,
                **_callbacks_args
            }
        }

    return unpack(callbacks_names=callbacks_names,
                  model=model,
                  workdir=workdir,
                  cross_validator=cross_validator,
                  scoring=scoring,
                  verbose=verbose,
                  **grid_args)


def build_callbacks(model,
                    mode,
                    workdir,
                    resumable,
                    scoring,
                    cross_validator,
                    early_stopping=None,
                    gradient_clipping=None,
                    lr_scheduler=None,
                    **kwargs):
    # Cross-validator:
    has_valid = is_cv_for_net(cross_validator)
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
                                        monitor=f"{monitor}_loss")))

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
    _score = ScoringWrapper(scoring)

    callbacks.append(
        ("score",
         EpochScoring(scoring=_score,
                      name=scoring,
                      on_train=not has_valid,
                      lower_is_better=not _score.greater_is_better)))

    # Callbacks names:
    callbacks_names = [c[0] for c in callbacks]

    return callbacks, callbacks_names


def build_callbacks_args(callbacks_names,
                         workdir,
                         ensure_list=False,
                         **kwargs):
    callbacks_args = filter_by_keys(kwargs, callbacks_names)
    return prefix_args("callbacks", ensure_list=ensure_list, **callbacks_args)


def __unpack_dataset(ds):
    from dataset import AslDataset

    if isinstance(ds, AslDataset):
        return ds
    elif hasattr(ds, 'dataset'):
        return __unpack_dataset(ds.dataset)
    else:
        return None


def format_dir(dir, **kwargs):
    return normpath(dir.format(**kwargs)) if dir is not None else ''


def filter_by_keys(map, keys_to_filter, not_in=False):
    return dict(
        filter(lambda o: not (o[0] in keys_to_filter) == not_in, map.items()))


def get_processed(dataset, field):
    data = [getattr(x, field) for x in dataset]
    return dataset.fields[field].process(data)


def get_cross_validator(cv, cv_args, seed, **kwargs):
    _cv = locate(cv)
    return _cv(random_state=seed, **cv_args)


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
    X, y = dataset.X(fmt="array"), dataset.y(fmt="array")
    original = Counter(y)

    # Compute samplings:
    under_sampling = compute_sampling(original, "under")
    over_sampling = compute_sampling(under_sampling, "over")

    # Prepare pipeline:
    rus = RandomUnderSampler(sampling_strategy=under_sampling,
                             random_state=seed)
    ros = RandomOverSampler(sampling_strategy=over_sampling, random_state=seed)
    pipeline = Pipeline(steps=[("under", rus), ("over", ros)])

    # Resample data:
    X_res, y_res = pipeline.fit_resample(X, y)
    dataset_res = AslDataset(dataset=dataset, data=(X_res, y_res))

    return dataset_res


class ScoringWrapper:
    def __init__(self, score_func):
        from sklearn.metrics import get_scorer
        self._score_func = score_func
        self.scorer = get_scorer(score_func)

    def __call__(self, estimator, X, y_true, sample_weight=None):
        if isinstance(y_true, AslSliceDataset):
            y_true = y_true.collated().cpu()
        return self.scorer(estimator, X, y_true, sample_weight)

    def __repr__(self):
        return f"{type(self).__name__}('{self._score_func}')"

    @property
    def greater_is_better(self):
        return (self.scorer._sign == 1)
