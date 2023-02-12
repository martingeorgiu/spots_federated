import argparse

import flwr as fl

from src.datamodule import HAM10000DataModule
from src.flower_client import FlowerClient
from src.model import MobileNetLightningModel


def client_fn(unit: int, no_units: int) -> FlowerClient:
    print("Creatign client...")
    print(f"unit: {unit}")
    print(f"no_units: {no_units}")

    # Model and data
    model = MobileNetLightningModel()

    dataModule = HAM10000DataModule(unit=unit, no_of_units=no_units)
    dataModule.setup("fit")

    # Flower client
    client = FlowerClient(
        model,
        dataModule.train_dataloader(),
        dataModule.val_dataloader(),
        dataModule.test_dataloader(),
    )
    return client


def main() -> None:
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-u", "--unit", help="unit index", type=int, default=0)
    argParser.add_argument("-nu", "--no_units", help="number of units", type=int, default=1)
    args = argParser.parse_args()

    client = client_fn(args.unit, args.no_units)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
