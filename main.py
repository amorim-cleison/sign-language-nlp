import pandas as pd
from commons.log import log
from commons.util import load_args, save_json
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier

import helper as h
from args import ARGUMENTS
from dataset import AslDataset


def run(args):
    # Seed:
    seed = args["seed"]
    h.setup_seed(seed)

    # Device:
    device = h.prepare_device(args["cuda"])

    # Dataset:
    # h.save_stats_datasets(device, args)
    dataset = AslDataset(device=device, batch_first=True, **args).stoi()

    if args["debug"]:
        dataset = dataset.truncated(1000)

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

    # Tune hyperparams and test:
    test_size = args["test_size"]
    test_data, train_data = dataset.split(lengths=test_size,
                                          indices_only=False,
                                          seed=seed)
    best_estimator = tune_hyperparams(estimator=net,
                                      callbacks_names=callbacks_names,
                                      train_data=train_data,
                                      **args)
    test_model(estimator=best_estimator, test_data=test_data, **args)


def tune_hyperparams(estimator, callbacks_names, train_data, **kwargs):
    log("\n==================== TUNING HYPERPARAMETERS ====================\n")

    # Grid search:
    grid_params = h.build_grid_params(callbacks_names=callbacks_names,
                                      data=train_data,
                                      **kwargs)
    gs = GridSearchCV(estimator=estimator, **grid_params)
    log(gs)

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
    log("Saving grid search output...")
    save_json(data=gs_output,
              path=f"{args['workdir']}/grid_search_output.json")
    pd.DataFrame(
        gs.cv_results_).to_csv(f"{args['workdir']}/grid_search_results.csv")

    # Return best estimator found:
    return gs.best_estimator_


def test_model(estimator, test_data, scoring, **kwargs):
    log("\n==================== TESTING MODEL ====================\n")

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
    log("Saving test output...")
    save_json(data=test_output, path=f"{args['workdir']}/test_output.json")


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
