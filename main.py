import pandas as pd
from commons.log import log
from commons.util import load_args, save_json
from sklearn.model_selection import GridSearchCV, cross_val_score
from skorch import NeuralNetClassifier

import helper as h
from args import ARGUMENTS
from dataset import AslDataset
from helper import ScoringWrapper


def run(args):
    args["workdir"] = h.format_dir(args["workdir"], **args)

    # Seed:
    h.setup_seed(**args)

    # Device:
    device = h.prepare_device(args["cuda"])

    # Dataset:
    dataset = AslDataset(device=device, batch_first=True, **args)

    if args["debug"]:
        dataset = dataset.truncated(1000)

    # Balance dataset:
    if should_balance_dataset(args):
        dataset = h.balance_dataset(dataset=dataset, seed=args["seed"])
    log(f"{len(dataset)} entries of data")

    # Cross-validator:
    # cross_validator = h.get_cross_validator(dataset=dataset, **args)
    from skorch.dataset import CVSplit
    cross_validator = CVSplit(5)

    # Callbacks:
    callbacks, callbacks_names = h.build_callbacks(
        cross_validator=cross_validator, **args, **args["training_args"])

    # Classifier:
    net_params = h.build_net_params(callbacks=callbacks,
                                    callbacks_names=callbacks_names,
                                    device=device,
                                    dataset=dataset,
                                    cross_validator=cross_validator,
                                    **args)

    # FIXME -------------------------------
    import torch.nn as nn
    import torch

    def collate(data):
        X, y = zip(*data)
        X, X_lengths = zip(*X)

        X = torch.stack(X)
        X_lengths = torch.stack(X_lengths)
        y = torch.stack(y)
        return {"X": X, "lengths": X_lengths, "y": y}, y

    net_params["iterator_train__collate_fn"] = collate
    net_params["iterator_valid__collate_fn"] = collate
    # ---------------------------------------

    net = NeuralNetClassifier(**net_params)

    # Train:
    if args["mode"] == "train":
        run_training(net=net,
                     dataset=dataset,
                     cross_validator=cross_validator,
                     **args,
                     **args["training_args"])

    # Grid search:
    elif args["mode"] == "grid":
        run_grid_search(net=net,
                        callbacks_names=callbacks_names,
                        dataset=dataset,
                        cross_validator=cross_validator,
                        **args)


def run_training(net, dataset, cross_validator, scoring, n_jobs, **kwargs):
    run_training_cv(net=net,
                    dataset=dataset,
                    cross_validator=cross_validator,
                    scoring=scoring,
                    n_jobs=n_jobs)


def run_training_cv(net, dataset, cross_validator, scoring, n_jobs):
    log(f"Training ({cross_validator})...")

    test, train = dataset.collated().split(0.2, indices_only=False)

    # Fit:
    net.fit(train, train.y().cpu())

    # Score:
    score = net.score(test, test.y().cpu())
    log(f"Test score: {score:.4f}")

    # Cross-validation:
    # scores = cross_val_score(net,
    #                          dataset.collated(),
    #                          dataset.collated().y(),
    #                          cv=5,
    #                          scoring=ScoringWrapper(scoring),
    #                          error_score='raise',
    #                          n_jobs=n_jobs)

    # log(f"'{scoring.capitalize()}': {[f'{x:.3f}' for x in scores]}")
    # log(f"AVG '{scoring}': {scores.mean():.3f}")


def run_grid_search(net, callbacks_names, dataset, cross_validator, **kwargs):
    log("Grid search...")

    # Grid search:
    grid_params = h.build_grid_params(callbacks_names=callbacks_names,
                                      cross_validator=cross_validator,
                                      **kwargs)
    gs = GridSearchCV(net, **grid_params)
    log(gs)

    # Fit:
    gs.fit(dataset.X(), dataset.y())

    # Output:
    gs_output = {
        "best_score": float(gs.best_score_),
        "best_params": gs.best_params_,
        "best_index": int(gs.best_index_),
        "scoring": str(gs.scoring)
    }
    log(gs_output)

    # Save output:
    log("Saving grid search output...")
    save_json(data=gs_output, path=f"{args['workdir']}/grid_search.json")
    pd.DataFrame(
        gs.cv_results_).to_csv(f"{args['workdir']}/grid_search_results.csv")


def should_balance_dataset(args):
    return ("balance_dataset" in args["dataset_args"]) and (
        args["dataset_args"]["balance_dataset"] is True)


if __name__ == "__main__":
    args = vars(load_args('SL Transformer', ARGUMENTS))
    run(args)
