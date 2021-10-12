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


def build_net_params(dataset_objs, training_args, model_args, dataset_args,
                     cuda, model, mode, resumable, callbacks, callbacks_names,
                     **kwargs):
    # Configure params:
    training_args["criterion"] = locate(training_args["criterion"])
    training_args["optimizer"] = locate(training_args["optimizer"])

    # Callbacks args:
    callbacks_args = build_callbacks_args(model=model,
                                          mode=mode,
                                          callbacks_names=callbacks_names,
                                          **training_args,
                                          **kwargs)

    # Module args:
    module_args = prefix_args("module",
                              ensure_list=False,
                              batch_first=True,
                              src_vocab=dataset_objs["src_vocab"],
                              tgt_vocab=dataset_objs["tgt_vocab"],
                              **model_args)

    # Other args:
    other_args = filter_by_keys(training_args,
                                keys_to_filter=callbacks_names,
                                not_in=True)

    # CV split:
    train_split = CVSplit(cv=dataset_args["samples_min_freq"], stratified=True)

    return {
        "device": prepare_device(cuda),
        "module": locate(model),
        "callbacks": callbacks,
        "train_split": train_split,
        **module_args,
        **callbacks_args,
        **other_args
    }


def build_grid_params(grid_args, dataset_args, callbacks_names, model, mode,
                      workdir, **kwargs):
    def unpack(training_args,
               dataset_args,
               model_args,
               callbacks_names,
               model,
               workdir,
               mode="grid",
               scoring="accuracy",
               verbose=3,
               **kwargs):
        # Callbacks args:
        callbacks_args = build_callbacks_args(model=model,
                                              mode=mode,
                                              workdir=workdir,
                                              callbacks_names=callbacks_names,
                                              ensure_list=True,
                                              **training_args)

        # Module args:
        module_args = prefix_args("module", ensure_list=True, **model_args)

        # Other args:
        other_args = filter_by_keys(training_args,
                                    keys_to_filter=callbacks_names,
                                    not_in=True)

        return {
            "refit": True,
            "cv": dataset_args["samples_min_freq"],
            "verbose": verbose,
            "scoring": scoring,
            "param_grid": {
                **module_args,
                **callbacks_args,
                **other_args
            }
        }

    return unpack(callbacks_names=callbacks_names,
                  dataset_args=dataset_args,
                  model=model,
                  mode=mode,
                  workdir=workdir,
                  **grid_args)


def build_callbacks(model, mode, workdir, resumable, early_stopping,
                    gradient_clipping, lr_scheduler, **kwargs):
    # Callbacks:
    callbacks = []

    # Checkpoint saving:
    checkpoint = Checkpoint(monitor='valid_loss_best', dirname=workdir)
    callbacks.append(("checkpoint", checkpoint))

    # Init state (resume):
    if resumable and mode == "train":
        callbacks.append(("resume_state", LoadInitState(checkpoint)))

    # Early stopping:
    callbacks.append(("early_stopping", EarlyStopping(**early_stopping)))

    # Gradient clip:
    callbacks.append(
        ("gradient_clipping", GradientNormClipping(**gradient_clipping)))

    # Progress bar (for epochs):
    callbacks.append(("progress_bar", ProgressBar()))

    # LR Scheduler:
    def lr_score(net, X=None, y=None):
        return net.optimizer_.param_groups[0]["lr"]

    callbacks.append(("lr_scoring", EpochScoring(lr_score, name='lr')))
    callbacks.append(("lr_scheduler",
                      LRScheduler(monitor="valid_loss",
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


def format_dir(dir, **kwargs):
    return normpath(dir.format(**kwargs)) if dir is not None else ''


def filter_by_keys(map, keys_to_filter, not_in=False):
    return dict(
        filter(lambda o: not (o[0] in keys_to_filter) == not_in, map.items()))


def get_processed(dataset, field):
    data = [getattr(x, field) for x in dataset]
    return dataset.fields[field].process(data)


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
