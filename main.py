from commons.util import load_args

from dataset.builder import DatasetBuilder
from model.builder import ModelBuilder
from runner import Runner
from args import ARGUMENTS

if __name__ == "__main__":
    args = vars(load_args('SL Transformer', ARGUMENTS))
    dataset_objs = DatasetBuilder().build(**args)
    model_objs = ModelBuilder(**dataset_objs).build(**args)
    Runner(**dataset_objs, **model_objs).run(**args)
