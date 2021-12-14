import pandas as pd
from commons.log import log
from commons.util import load_args, save_json
from sklearn.model_selection import GridSearchCV, cross_val_score
from skorch import NeuralNetClassifier

import helper as h
from args import ARGUMENTS
from dataset import AslDataset


def run(args):
    # Workdir:
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

    # Callbacks:
    callbacks, callbacks_names = h.build_callbacks(**args,
                                                   **args["training_args"])

    # Classifier:
    net_params = h.build_net_params(callbacks=callbacks,
                                    callbacks_names=callbacks_names,
                                    device=device,
                                    dataset=dataset,
                                    **args)
    net = NeuralNetClassifier(**net_params)

    # Train:
    if args["mode"] == "train":
        run_training(net=net, dataset=dataset, **args, **args["training_args"])

    # Grid search:
    elif args["mode"] == "grid":
        run_grid_search(net=net,
                        callbacks_names=callbacks_names,
                        dataset=dataset,
                        **args)


def run_training(net, dataset, test_size, **kwargs):
    log("Training...")

    dataset = dataset.stoi()
    test, train = dataset.split(test_size, indices_only=False)

    # Fit:
    net.fit(train, train.y().cpu())

    # Score:
    score = net.score(test, test.y().cpu())
    log(f"Test score: {score:.4f}")


def run_grid_search(net, callbacks_names, dataset, **kwargs):
    log("Grid search...")

    # Grid search:
    grid_params = h.build_grid_params(callbacks_names=callbacks_names,
                                      **kwargs)
    gs = GridSearchCV(net, **grid_params)
    log(gs)

    # Fit:
    dataset = dataset.stoi()
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
