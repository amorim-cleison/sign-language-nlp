# A Linguistic Approach to Sign Language Recognition (Sign Language NLP)

![Build](https://github.com/amorim-cleison/sl-transformer/workflows/Build/badge.svg)
![Code Quality](https://github.com/amorim-cleison/sl-transformer/workflows/Code%20Quality/badge.svg)

This work aims at developing an approach to Sign Language Recognition that is centered on its linguistics, and applies techniques from Natural Language Processing (NLP) that are proven to be successful in another language-related tasks (like speech recognition, handwritten text, among others).


## Initialization

This project requires [Python](https://www.python.org/downloads/) 3.8 or higher and [Poetry](https://python-poetry.org/). Poetry is a dependency management tool that will help you to properly manage dependencies needed here. 

Once you have installed Python, install Poetry executing this command:

```bash
pip install poetry
```

Then, ask Poetry to the project dependencies executing this command:

```bash
poetry install
```

After a couple of minutes, you might be ready to use this project in your machine.

> Poetry might ask you to download the `commons-python` dependency source code. 
If that happens, just clone it from the URL https://github.com/amorim-cleison/commons-python.


## Download the ASL-Phono

This source code was designed to run properly with the **ASL-Phono** dataset. You can download it easily by following the link below:

* [ASL-Phono download](https://www.cin.ufpe.br/~cca5/asl-phono/download)

Once downloaded, you can save it at the path below in your local machine -- that's just a suggestion, feel free to customize it and modify the configurations accordingly.
```
~/work/dataset/asl-phono
```

You can also learn further about the dataset by taking a look at the following resources:
* [Repository](https://www.cin.ufpe.br/~cca5/asl-phono/)
* [Paper](http://www.cin.ufpe.br/~cca5/asl-datasets/paper)


## Configure the project execution

There are some configuration files available in the [/config](./config/) folder that you can take as reference to duplicate and customize your execution.

See below an example `config` file that executes a full grid search to find the best hyper-parameters set for a Transformer model and, once found, tests the best trained model and logs the accuracy and other metrics about its performance.

```yaml
# Example 'config' file:

debug: False
cuda: True
seed: 1
workdir: '../../work/sl-transformer/checkpoint/{model}/{datetime:%Y-%m-%d-%H-%M-%S}'
verbose: 3
n_jobs: -1
cv: 5
lr:  # tuned in grid search
scoring: [neg_log_loss, accuracy, precision_weighted, recall_weighted, f1_weighted]
max_epochs: 200
batch_size: 50
test_size: 0.15

early_stopping:
  patience: 30
  threshold: 1e-4
  threshold_mode: rel

gradient_clipping:
  gradient_clip_value: 0.5

lr_scheduler: 
  policy: ReduceLROnPlateau
  factor: 0.2  # factor of 5 (1/5)
  patience: 5

# Model:
model: model.EncoderDecoderLSTMAttn
model_args:
  embedding_size: # tuned in grid search
  hidden_size:    # tuned in grid search
  num_layers:     # tuned in grid search
  dropout:        # tuned in grid search

# Criterion:
criterion: torch.nn.CrossEntropyLoss

# Optimizer:
optimizer: torch.optim.SGD
optimizer_args:
  nesterov: False
  momentum: 0.9

# Grid search:
grid_args:
  lr: [0.1, 0.01, 0.001]
  model_args:
    embedding_size: [1024, 512, 128]
    hidden_size: [512, 256, 128]
    num_layers: [6, 4, 2]
    dropout: [0.5, 0.1]

# Dataset:
dataset_args:
  dataset_dir: ../../work/dataset/asl-phono/phonology/3d
  fields: [
    orientation_dh,
    orientation_ndh, 
    movement_dh, 
    movement_ndh,
    handshape_dh, 
    handshape_ndh, 
    # mouth_openness
  ]
  samples_min_freq: 2   # How many samples for the label should exist in dataset for the label to be considered?
  composition_strategy: as_words
  reuse_transient: False
  balance_dataset: True
```

## Execute the project

Once the execution is configured, you can easily run it in your local machine by using the following command:

```bash
# Execute the project:
poetry run python main.py --config '<your_config_file.yaml>' --n_jobs=-1
```

### Using Dask clusters

If you have a [Dask](https://www.dask.org/) cluster available, you can leverage it and take great improvements in your project execution. Just provide its scheduler address, as follows:

```bash
# Execute the project in a 'Dask' cluster
poetry run python main.py --config '<your_config_file.yaml>' --n_jobs=-1 --dask "{ 'scheduler': '<address_or_ip:port>', 'source': '<path_to_this_project_code>.zip'}"
```

### Using other clusters

If you're using other types of clusters or workload managers, take a look at the files in the [cluster](./cluster/) folder to find some additional inspiration. Also, feel free to collaborate with additional examples.

### Using CUDA GPUs
You can improve even further your project execution if you have CUDA GPUs available. Just make sure to have the following environment variable set with the GPUs indexes to be used before running the commands above:

```bash
# Set the environment variable below with the GPUs available (e.g.: 0,1,2):
CUDA_VISIBLE_DEVICES=<available_gpus_indexes>
```


## Contact
If you have questions, comments or contributions, please contact me at:

```md
Cleison Amorim  : cca5@cin.ufpe.br
```