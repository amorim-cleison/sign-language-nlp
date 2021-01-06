# from example.first_example import run
# from example.real_world_example import run
from sl_transformer import run
from commons.util import Argument, load_args


ARGUMENTS = [
    Argument('-ad', '--attributes_dir', type=str, required=True, help='Attributes directory'),
    Argument('-dp', '--dataset_path', type=str, required=True, help='Dataset path'),
    Argument('-d', '--debug', type=bool, help='Debug flag'),
]

if __name__ == "__main__":
    args = load_args('SL Transformer', ARGUMENTS)
    run(**vars(args))
