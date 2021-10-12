from commons.util import Argument

ARGUMENTS = [
    Argument('-m', '--model', help='Model class'),
    Argument('-n', '--mode', options=["grid", "train"], help='Mode'),
    Argument('-r', '--resumable', type=bool, help='Resume tasks'),
    Argument('-w', '--workdir', help='Working directory'),
    Argument('-d', '--debug', type=bool, default=False, help='Debug flag'),
    Argument('-nv', '--cuda', type=bool, default=False, help='Enable cuda'),
    Argument('-k', '--seed', type=int, required=True, help='Seed'),
    Argument('-ds',
             '--dataset_args',
             type=dict,
             help='Options for the dataset'),
    Argument('-md', '--model_args', type=dict, help='Options for the model'),
    Argument('-gr', '--grid_args', type=dict, help='Options for the grid search'),
    Argument('-tr',
             '--training_args',
             type=dict,
             help='Options for the training'),
]
