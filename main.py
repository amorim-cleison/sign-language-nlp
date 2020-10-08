# from example.first_example import run
# from example.real_world_example import run
from sl_transformer import run

if __name__ == "__main__":
    args = {
        "input_dir": "../../work/sl-phono-parser/attributes/"
    }
    run(**args)
