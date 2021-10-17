import pandas as pd
from commons.log import log
from commons.util import load_args, save_json
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
from skorch.helper import SliceDataset

import helper as h
from args import ARGUMENTS
from dataset import AslDataset


def run(args):
    args["workdir"] = h.format_dir(args["workdir"], **args)

    # Seed:
    h.setup_seed(**args)

    # Device:
    device = h.prepare_device(args["cuda"])

    # Dataset:
    dataset = AslDataset(device=device,
                         batch_first=True,
                         debug=args["debug"],
                         **args["dataset_args"])

    # Callbacks:
    callbacks, callbacks_names = h.build_callbacks(model=args["model"],
                                                   mode=args["mode"],
                                                   workdir=args["workdir"],
                                                   resumable=args["resumable"],
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
        net.fit(dataset, None)

    # Grid search:
    elif args["mode"] == "grid":
        run_grid_search(net=net,
                        callbacks_names=callbacks_names,
                        dataset=dataset)


def run_grid_search(net, callbacks_names, dataset):
    # Slice data (sklearn is not compatible with 'dataset'):
    X = SliceDataset(dataset, idx=0)
    y = SliceDataset(dataset, idx=1)

    # Grid search:
    grid_params = h.build_grid_params(callbacks_names=callbacks_names, **args)
    gs = GridSearchCV(net, **grid_params)
    gs.fit(X, y)

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
