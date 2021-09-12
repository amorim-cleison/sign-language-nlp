from commons.util import Argument

ARGUMENTS = [
    Argument('-mt', '--model_type', help='Model type'),
    Argument('-d', '--debug', type=bool, default=False, help='Debug flag'),
    Argument('-s', '--seed', type=int, help='Random seed for reproducibility'),
    Argument('-nv', '--cuda', type=bool, default=False, help='Enable cuda'),
    Argument('-ds',
             '--dataset_args',
             type=dict,
             help='Options for the dataset'),
    Argument('-md', '--model_args', type=dict, help='Options for the model'),
    Argument('-tr',
             '--training_args',
             type=dict,
             help='Options for the training'),
    Argument('-tl',
             '--transfer_learning',
             type=dict,
             help='Options for transfering learning'),
]
