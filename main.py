import pandas as pd
from commons.log import log
from commons.util import load_args, save_json
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier

import helper as h
from args import ARGUMENTS
from dataset.builder import DatasetBuilder


def run(args):
    dataset_objs = DatasetBuilder().build(**args)
    args["workdir"] = h.format_dir(args["workdir"], **args)

    # Dataset:
    ds = dataset_objs["dataset"]
    X, X_lens = h.get_processed(ds, "src")
    y = h.get_processed(ds, "tgt").squeeze()

    # Seed:
    h.setup_seed(**args)

    # Callbacks:
    callbacks, callbacks_names = h.build_callbacks(model=args["model"],
                                                   mode=args["mode"],
                                                   workdir=args["workdir"],
                                                   resumable=args["resumable"],
                                                   **args["training_args"])

    # Classifier:
    net_params = h.build_net_params(dataset_objs=dataset_objs,
                                    callbacks=callbacks,
                                    callbacks_names=callbacks_names,
                                    **args)
    net = NeuralNetClassifier(**net_params)

    # Train:
    if args["mode"] == "train":
        net.fit({"X": X.permute(1, 0), "lengths": X_lens}, y)

    # Grid search:
    if args["mode"] == "grid":
        run_grid_search(net=net, callbacks_names=callbacks_names, X=X, y=y)


def run_grid_search(net, callbacks_names, X, y):
    grid_params = h.build_grid_params(callbacks_names=callbacks_names, **args)
    gs = GridSearchCV(net, **grid_params)
    gs.fit(X.permute(1, 0), y)

    # Output:
    gs_output = {
        "best_score": float(gs.best_score_),
        "best_params": gs.best_params_,
        "best_index": int(gs.best_index_),
        "scoring": gs.scoring
    }
    print(gs_output)

    # Save output:
    log("Saving grid search output...")
    save_json(data=gs_output, path=f"{args['workdir']}/grid_search.json")
    pd.DataFrame(
        gs.cv_results_).to_csv(f"{args['workdir']}/grid_search_results.csv")


if __name__ == "__main__":
    args = vars(load_args('SL Transformer', ARGUMENTS))
    run(args)
