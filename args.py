from commons.util import Argument

ARGUMENTS = [
    Argument('-m', '--model', help='Model class'),
    Argument('-o', '--optimizer', help='Optimizer class'),
    Argument('-f', '--criterion', help='Criterion class'),
    Argument('-cv', '--cv', help='Cross-validator class'),
    Argument('-sc', '--scoring', type=str, help='Scoring metric to use'),
    Argument('-vb', '--verbose', type=int, help='Verbosity level'),
    Argument('-j', '--n_jobs', type=int, default=1, help='Number of jobs'),
    Argument('-n', '--mode', options=["grid", "train"], help='Mode'),
    Argument('-w', '--workdir', help='Working directory'),
    Argument('-d', '--debug', type=bool, default=False, help='Debug flag'),
    Argument('-nv', '--cuda', type=bool, default=False, help='Enable cuda'),
    Argument('-k', '--seed', type=int, required=True, help='Seed'),
    Argument('-lr', '--lr', type=float, required=True, help='Learning rate'),
    Argument('-ep', '--max_epochs', type=int, required=True,
             help='Max epochs'),
    Argument('-bs', '--batch_size', type=int, required=True,
             help='Batch size'),
    Argument('-ts', '--test_size', type=float, required=True,
             help='Test size'),
    Argument('-es',
             '--early_stopping',
             type=dict,
             help='Options for early stopping'),
    Argument('-gcl',
             '--gradient_clipping',
             type=dict,
             help='Options for gradient clipping'),
    Argument('-lrs',
             '--lr_scheduler',
             type=dict,
             help='Options for learning rate scheduler'),
    Argument('-ds',
             '--dataset_args',
             type=dict,
             help='Options for the dataset'),
    Argument('-ma', '--model_args', type=dict, help='Options for the model'),
    Argument('-oa',
             '--optimizer_args',
             type=dict,
             help='Options for the optimizer'),
    Argument('-ca',
             '--criterion_args',
             type=dict,
             help='Options for the criterion'),
    Argument('-cva',
             '--cv_args',
             type=dict,
             help='Options for the cross-validator'),
    Argument('-gr',
             '--grid_args',
             type=dict,
             help='Options for the grid search'),
]
