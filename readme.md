# Federated learning of ANNs for image classification

## Setup

Download the dataset from [here](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), rename the dictionary to `dataset` and move it to the root of this project.

## Usage

You can run following `./bin/run-proper-flower.sh` to run a real local federation with 3 clients using FedAvg with 50 rounds.

Example command to run local simulation of federated learning using FedAvg:

`python server.py --simulated --rounds 50 --strategy fedavg --train_epochs 3`

For additional documentation of all features of this example repo, feel free to dive into any of the root python files which all can be individually run. Also for simple docs you can add `--help` flag for basic explanation of the scripts.

## Logging

To browse the logs use command `tensorboard --logdir=dir_xy` and replace `dir_xy` accordingly:

- `classic-models/` - logs from non federated trainings will be logged there
- `federated-models/` - logs from federated trainings without using any simulated clients
- `federated-models-simulated/` - logs from federated trainings with the use of simulated clients
