# from sl_transformer import run
# from v2.sl_transformer import run
# from v2.sl_rnn import run
from v2 import ModelRunner, ModelBuilder
from commons.util import Argument, load_args

ARGUMENTS = [
    Argument('-mt', '--model_type', help='Model type'),
    Argument('-d', '--debug', type=bool, default=False, help='Debug flag'),
    Argument('-s', '--seed', type=int, help='Random seed for reproducibility'),
    Argument('-nv', '--cuda', type=bool, default=False, help='Enable cuda'),
    Argument('-ds', '--dataset', type=dict, help='Options for the dataset'),
    Argument('-md', '--model', type=dict, help='Options for the model'),
    Argument('-tr', '--training', type=dict, help='Options for the training'),
    Argument('-tl',
             '--transfer_learning',
             type=dict,
             help='Options for transfering learning'),
]

if __name__ == "__main__":
    args = load_args('SL Transformer', ARGUMENTS)
    args = {
        f"{k}_args" if isinstance(v, dict) else k: v
        for (k, v) in vars(args).items()
    }

    objects_built = ModelBuilder().build(**args)

    ModelRunner(**objects_built).run(**args)
