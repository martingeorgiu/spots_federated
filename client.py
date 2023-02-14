import argparse
from typing import Union

import flwr as fl
import torch

from src.consts import get_alpha
from src.datamodule import HAM10000DataModule
from src.flower_client import FlowerClient
from src.model import MobileNetLightningModel


def client_fn(
    unit: int,
    no_units: int,
    alpha: Union[torch.FloatTensor, None],
    minified: bool = False,
    train_epochs: int = 1,
    gamma: Union[int, float] = 2,
) -> FlowerClient:
    print("Creatign client...")
    print(f"unit: {unit}")
    print(f"no_units: {no_units}")
    print(f"train_epochs: {train_epochs}")
    print(f"alpha: {alpha}")
    print(f"gamma: {gamma}")

    # Model and data
    model = MobileNetLightningModel(alpha=alpha, gamma=gamma)

    dataModule = HAM10000DataModule(unit=unit, no_of_units=no_units, minified=minified)
    dataModule.setup("fit")

    # Flower client
    client = FlowerClient(
        model,
        dataModule.train_dataloader(),
        dataModule.val_dataloader(),
        dataModule.test_dataloader(),
        train_epochs=train_epochs,
    )
    return client


def main() -> None:
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-u", "--unit", help="unit index", type=int, default=0)
    argParser.add_argument("-nu", "--no_units", help="number of units", type=int, default=1)
    argParser.add_argument(
        "-m",
        "--minified",
        help="only applicable in simulated mode, run with minified dataset",
        action=argparse.BooleanOptionalAction,
    )
    argParser.add_argument(
        "-te", "--train_epochs", help="number of train epochs per round", type=int, default=1
    )
    argParser.add_argument("-a", "--alpha", help="alpha parameter of focal loss", type=str)
    argParser.add_argument(
        "-g", "--gamma", help="gamma parameter of focal loss", type=float, default=2
    )
    args = argParser.parse_args()
    alpha = get_alpha(args.alpha)

    client = client_fn(
        unit=args.unit,
        no_units=args.no_units,
        minified=args.minified == True,
        train_epochs=args.train_epochs,
        alpha=alpha,
        gamma=args.gamma,
    )
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
