import flwr as fl
import pytorch_lightning as pl
from collections import OrderedDict
import torch

from src.datamodule import HAM10000DataModule
from src.model import MobileNetLightningModel
from torch.utils.data import DataLoader,Dataset


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        features_params = get_parameters_from_model(self.model.model.features)
        classifier_params = get_parameters_from_model(self.model.model.classifier)

        return features_params+classifier_params
        

    def set_parameters(self, parameters):
        set_parameters_on_model(self.model.model.features, parameters[0:240])
        set_parameters_on_model(self.model.model.classifier, parameters[240:244])

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(self.model, self.train_loader, self.val_loader)
        
        return self.get_parameters(config={}), len(self.train_loader.dataset)+len(self.val_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer()
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]

        return loss, len(self.test_loader.dataset), {"loss": loss}


def get_parameters_from_model(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters_on_model(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    # Model and data
    model = MobileNetLightningModel()

    dataModule  = HAM10000DataModule()
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