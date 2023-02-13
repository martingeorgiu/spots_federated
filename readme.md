# spots distributed

## Usage

Download the dataset from [here](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), rename the dictionary to `dataset` and move it to the root of this project.

You can run following `./bin/run-proper-flower.sh` to run a real local federation with 3 clients using FedAvg with 50 rounds.

Example command to run local simulation of federated learning using FedAvg:

`python server.py --simulated --rounds 50 --strategy fedavg --train_epochs 3`

## TODO

- parametrized loss function even with it's params (like weights), using argparse
- weight decay
- momentum
- chytrejsi rozdeleni datasetu aby vahy byly stejne
- lepsi normalizace
- nejake vic kompenzace nerovnomerneho datasetu?
- mozne dalsi pokusy
  - oversampling a trochu undersampling
