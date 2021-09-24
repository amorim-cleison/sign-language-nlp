from commons.util import load_args

from dataset.builder import DatasetBuilder
from model.builder import ModelBuilder
from runner import Runner
from args import ARGUMENTS
from ray import tune


def execute(config):
    print(config)
    tune.report(mean_loss=1.06)
    # model_objs = ModelBuilder(**dataset_objs).build(**args)
    # Runner(**dataset_objs, **model_objs).run(**args)
    return {"acc": 0.6, "loss": 1.06}


if __name__ == "__main__":
    args = vars(load_args('SL Transformer', ARGUMENTS))
    # dataset_objs = DatasetBuilder().build(**args)

    analysis = tune.run(execute,
                        config={"lr": tune.grid_search([0.001, 0.01, 0.1])})

    print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

    # Get a dataframe for analyzing trial results.
    df = analysis.dataframe()
    print(df)
