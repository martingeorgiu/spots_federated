import argparse

import flwr as fl

from src.datamodule import HAM10000DataModule
from src.flower_client import FlowerClient
from src.model import MobileNetLightningModel


def client_fn(
    unit: int, no_units: int, minified: bool = False, train_epochs: int = 1
) -> FlowerClient:
    print("Creatign client...")
    print(f"unit: {unit}")
    print(f"no_units: {no_units}")
    print(f"train_epochs: {train_epochs}")

    # Model and data
    model = MobileNetLightningModel()

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
    args = argParser.parse_args()

    client = client_fn(args.unit, args.no_units, args.minified == True, args.train_epochs)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
