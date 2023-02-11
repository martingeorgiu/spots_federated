import flwr as fl
from src.flower_client import FlowerClient

from src.datamodule import HAM10000DataModule
from src.model import MobileNetLightningModel
import argparse

def main() -> None:
  argParser = argparse.ArgumentParser()
  argParser.add_argument("-u", "--unit", help="unit index", type=int, default=0)
  argParser.add_argument("-nu", "--no_units", help="number of units", type=int, default=1)
  args = argParser.parse_args()

  # Model and data
  model = MobileNetLightningModel()

  dataModule  = HAM10000DataModule(unit=args.unit, no_of_units=args.no_units)
  dataModule.setup('fit')

  # Flower client
  client = FlowerClient(model, 
  dataModule.train_dataloader(),
  dataModule.val_dataloader(),
  dataModule.test_dataloader(),
  )
  fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
  main()