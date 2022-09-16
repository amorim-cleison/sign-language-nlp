import pandas as pd
from commons.log import log
from commons.util import load_args
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier

import helper as h
from args import ARGUMENTS
from dataset import AslDataset
import torch


def run(args):
    # Seed:
    seed = args["seed"]
    h.setup_seed(seed)

    # Device:
    cuda = args["cuda"]
    device = h.prepare_device(cuda)

    # Dataset:
    # h.save_stats_datasets(device, args)
    dataset = AslDataset(device=device, batch_first=True, **args).stoi()

    if args["debug"]:
        dataset = dataset.truncated(100)

    # Balance dataset:
    if should_balance_dataset(args):
        dataset = h.balance_dataset(dataset=dataset, seed=seed)
    log(f"{len(dataset)} entries of data")

    # Callbacks:
    callbacks, callbacks_names = h.build_callbacks(dataset=dataset, **args)

    # Classifier:
    net_params = h.build_net_params(callbacks=callbacks,
                                    callbacks_names=callbacks_names,
                                    device=device,
                                    dataset=dataset,
                                    **args)
    net = NeuralNetClassifier(**net_params)

    # Split data:
    test_size = args["test_size"]
    test_data, train_data = dataset.split(lengths=test_size,
                                          indices_only=False,
                                          seed=seed)

    # Tune hyperparams and test model:
    best_estimator = tune_hyperparams(estimator=net,
                                      callbacks_names=callbacks_names,
                                      train_data=train_data,
                                      **args)
    test_model(estimator=best_estimator, test_data=test_data, **args)


def tune_hyperparams(estimator,
                     callbacks_names,
                     train_data,
                     cuda,
                     profile=True,
                     **kwargs):
    log("\n==================== TUNING HYPERPARAMETERS ====================\n")
    phase = "grid_search"

    with torch.autograd.profiler.profile(enabled=profile,
                                         use_cuda=cuda,
                                         profile_memory=True,
                                         with_flops=True) as prof:
        # Grid search:
        grid_params = h.build_grid_params(callbacks_names=callbacks_names,
                                          data=train_data,
                                          **kwargs)
        gs = GridSearchCV(estimator=estimator, **grid_params)
        log(grid_params)

        # Fit:
        gs.fit(X=train_data.X(), y=train_data.y().to_array())

        # Output:
        gs_output = {
            "best_score": float(gs.best_score_),
            "best_params": gs.best_params_,
            "best_index": int(gs.best_index_),
            "scoring": str(gs.scoring)
        }
        log(gs_output)

    # Save output:
    log("Saving output...")
    h.save_output(gs_output, phase=phase, **kwargs)
    pd.DataFrame(
        gs.cv_results_).to_csv(f"{args['workdir']}/{phase}_results.csv")

    # Save profile:
    log("Saving profile...")
    h.save_profile(prof, phase=phase, **kwargs)

    # Return best estimator found:
    return gs.best_estimator_


def test_model(estimator, test_data, scoring, cuda, profile=True, **kwargs):
    log("\n==================== TESTING MODEL ====================\n")
    phase = "test"

    with torch.autograd.profiler.profile(enabled=profile,
                                         use_cuda=cuda,
                                         profile_memory=True,
                                         with_flops=True) as prof:
        # Metrics:
        scorers = h.build_scoring(scoring=["accuracy", *scoring],
                                  labels=test_data.labels())
        test_output = {
            f"test_{scorer.score}": scorer(estimator, test_data.X(),
                                           test_data.y().to_array())
            for scorer in scorers
        }
        log(test_output)

    # Save output:
    log("Saving output...")
    h.save_output(test_output, phase=phase, **kwargs)

    # Save profile:
    log("Saving profile...")
    h.save_profile(prof, phase=phase, **kwargs)


def should_balance_dataset(args):
    return ("balance_dataset" in args["dataset_args"]) and (
        args["dataset_args"]["balance_dataset"] is True)


if __name__ == "__main__":
    args = vars(load_args('SL Transformer', ARGUMENTS))
    args["workdir"] = h.format_dir(args["workdir"], **args)

    # Dump args:
    h.dump_args(args)

    # Run:
    run(args)
