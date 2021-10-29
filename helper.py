from pydoc import locate

import torch
from commons.log import log
from commons.util import normpath
from skorch.callbacks import (Checkpoint, EarlyStopping, EpochScoring,
                              GradientNormClipping, LoadInitState, LRScheduler,
                              ProgressBar)
from skorch.dataset import CVSplit


def setup_seed(seed, **kwargs):
    torch.manual_seed(seed)


def prepare_device(cuda):
    if torch.cuda.is_available():
        if not cuda:
            log("WARNING: You have a CUDA device, so you should probably "
                "run with --cuda")

    return torch.device("cuda" if cuda else "cpu")


def build_net_params(training_args, model_args, model, optimizer, criterion,
                     mode, resumable, callbacks, callbacks_names, device,
                     dataset, optimizer_args, criterion_args, **kwargs):
    # Configure params:
    training_args["train_split"] = get_train_split(**training_args)

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
                               src_vocab=dataset.src_vocab,
                               tgt_vocab=dataset.tgt_vocab,
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
    other_args = filter_by_keys(training_args,
                                keys_to_filter=callbacks_names,
                                not_in=True)

    return {
        "device": device,
        "module": locate(model),
        "optimizer": locate(optimizer),
        "criterion": locate(criterion),
        "callbacks": callbacks,
        **_module_args,
        **_optimizer_args,
        **_criterion_args,
        **_callbacks_args,
        **_iterator_train_args,
        **_iterator_valid_args,
        **other_args
    }


def build_grid_params(grid_args, net_train_split, X, y, callbacks_names, model,
                      mode, workdir, **kwargs):
    def unpack(training_args,
               callbacks_names,
               model,
               workdir,
               X,
               y,
               net_train_split=None,
               mode="grid",
               scoring="accuracy",
               n_jobs=None,
               verbose=3,
               model_args={},
               optimizer_args={},
               criterion_args={},
               **kwargs):
        # Callbacks args:
        _callbacks_args = build_callbacks_args(model=model,
                                               mode=mode,
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
        other_args = filter_by_keys(training_args,
                                    keys_to_filter=callbacks_names,
                                    not_in=True)

        # CV:
        if ("train_split" in grid_args):
            train_split = get_train_split(**grid_args)
        elif net_train_split:
            train_split = net_train_split
        else:
            train_split = get_train_split(**grid_args)

        train, valid = train_split(X, get_collated_y(y))
        cv = [(train.indices, valid.indices)]

        # Scoring:
        _scoring = ScoringWrapper(scoring)

        return {
            "refit": True,
            "cv": cv,
            "verbose": verbose,
            "scoring": _scoring,
            "n_jobs": n_jobs,
            "error_score": "raise",
            "param_grid": {
                **_module_args,
                **_optimizer_args,
                **_criterion_args,
                **_callbacks_args,
                **other_args
            }
        }

    return unpack(callbacks_names=callbacks_names,
                  model=model,
                  mode=mode,
                  workdir=workdir,
                  net_train_split=net_train_split,
                  X=X,
                  y=y,
                  **grid_args)


def build_callbacks(model,
                    mode,
                    workdir,
                    resumable,
                    train_split=None,
                    early_stopping=None,
                    gradient_clipping=None,
                    lr_scheduler=None,
                    **kwargs):
    has_valid = train_split is not None

    # Callbacks:
    callbacks = []

    # Checkpoint saving:
    checkpoint = Checkpoint(
        monitor="valid_loss_best" if has_valid else "train_loss_best",
        dirname=workdir)
    callbacks.append(("checkpoint", checkpoint))

    # Init state (resume):
    if resumable and mode == "train":
        callbacks.append(("resume_state", LoadInitState(checkpoint)))

    # Early stopping:
    if early_stopping:
        callbacks.append(("early_stopping", EarlyStopping(**early_stopping)))

    # Gradient clip:
    if gradient_clipping:
        callbacks.append(
            ("gradient_clipping", GradientNormClipping(**gradient_clipping)))

    # Progress bar (for epochs):
    # callbacks.append(("progress_bar", ProgressBar()))

    # LR Scheduler:
    if lr_scheduler:

        def lr_score(net, X=None, y=None):
            return net.optimizer_.param_groups[0]["lr"]

        callbacks.append(("lr_scoring", EpochScoring(lr_score, name='lr')))
        callbacks.append(
            ("lr_scheduler",
             LRScheduler(monitor="valid_loss" if has_valid else "train_loss",
                         step_every="epoch",
                         **lr_scheduler)))

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


def get_collated_y(y):
    ds = __unpack_dataset(y)

    if ds is not None:
        y = ds.collate_target(y).cpu().numpy()
    return y


def format_dir(dir, **kwargs):
    return normpath(dir.format(**kwargs)) if dir is not None else ''


def filter_by_keys(map, keys_to_filter, not_in=False):
    return dict(
        filter(lambda o: not (o[0] in keys_to_filter) == not_in, map.items()))


def get_processed(dataset, field):
    data = [getattr(x, field) for x in dataset]
    return dataset.fields[field].process(data)


def get_train_split(train_split=None, **kwargs):
    if train_split:
        return CVSplit(**train_split)
    return train_split


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


class ScoringWrapper:
    def __init__(self, score_func):
        from sklearn.metrics import get_scorer
        self._score_func = score_func
        self.scorer = get_scorer(score_func)

    def __call__(self, estimator, X, y_true, sample_weight=None):
        y_true = get_collated_y(y_true)
        return self.scorer(estimator, X, y_true, sample_weight)

    def __repr__(self):
        return f"{type(self).__name__}('{self._score_func}')"
