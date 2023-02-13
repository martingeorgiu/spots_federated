import os
import shutil
import uuid
from collections import OrderedDict

import flwr as fl
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        train_epochs: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_epochs = train_epochs

    def get_parameters(self, config):
        features_params = get_parameters_from_model(self.model.model.features)
        classifier_params = get_parameters_from_model(self.model.model.classifier)

        return features_params + classifier_params

    def set_parameters(self, parameters):
        set_parameters_on_model(self.model.model.features, parameters[0:240])
        set_parameters_on_model(self.model.model.classifier, parameters[240:244])

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        id = str(uuid.uuid4())

        # Utility for logging metrics somewher to then read them back and return
        logger = CSVLogger("temp", name="fit", version=id)
        trainer = pl.Trainer(
            max_epochs=self.train_epochs, logger=logger, enable_checkpointing=False
        )
        trainer.fit(self.model, self.train_loader, self.val_loader)

        path = os.path.join("temp", "fit", id)
        df = pd.read_csv(os.path.join(path, "metrics.csv"))
        metrics = get_train_acc(df) | get_val_dict(df)
        shutil.rmtree(path)
        print(f"Flower client fit metrics: {metrics}")
        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset) + len(self.val_loader.dataset),
            metrics,
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(logger=False, enable_checkpointing=False)
        results = trainer.test(self.model, self.test_loader)
        test_step_log_dict = results[0]
        loss = test_step_log_dict["test_loss"]

        return loss, len(self.test_loader.dataset), {"eval_acc": test_step_log_dict["test_acc"]}


def get_parameters_from_model(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters_on_model(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_train_acc(df: pd.DataFrame) -> dict:
    train_dict = {}
    if "train_acc_epoch" in df:
        train_dict["train_acc"] = df["train_acc_epoch"].mean()
    return train_dict


def get_val_dict(df: pd.DataFrame) -> dict:
    val_dict = {}
    if "val_loss" in df:
        val_dict["val_loss"] = df["val_loss"].mean()
    if "val_acc" in df:
        val_dict["val_acc"] = df["val_acc"].mean()
    return val_dict
