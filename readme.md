# spots distributed

## Usage

Download the dataset from [here](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), rename the dictionary to `dataset` and move it to the root of this project.

You can run following `./bin/run-proper-flower.sh` to run a real local federation with 3 clients using FedAvg with 50 rounds.

Example command to run local simulation of federated learning using FedAvg:

`python server.py --simulated --rounds 50 --strategy fedavg --train_epochs 3`

## TODO

- use checkpoint instead of npy (what exactly is the difference?)
- weight decay
- momentum
- better normalization
- other compensations of imbalanced dataset
  - oversampling a undersampling
