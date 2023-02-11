import time
import os
import pytorch_lightning as pl
import torch

from torch import Tensor
from src.flower_client import set_parameters_on_model

from src.datamodule import HAM10000DataModule,lesion_type_dict
from src.model import MobileNetLightningModel

from glob import glob 
from torchvision import transforms
from PIL import Image
from pytorch_lightning.loggers import CSVLogger
from torchvision import models
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from random import randrange
import argparse

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--path", help="path of the stored model", type=str, required=True)
    argParser.add_argument("-f", "--federated", help="use federated storage format", action=argparse.BooleanOptionalAction)
    args = argParser.parse_args()

    def getModel() -> MobileNetLightningModel:
        if args.federated:
            print("Loading federated model")
            model = MobileNetLightningModel()
            arrays =  np.load(args.path, allow_pickle=True)
            set_parameters_on_model(model.model.features, arrays[0:240])
            set_parameters_on_model(model.model.classifier, arrays[240:244])
            return model
        else: 
            print("Loading classic model")
            return MobileNetLightningModel.load_from_checkpoint(args.path, map_location=torch.device('cpu'))

    model = getModel()
    # {'test_acc': 0.8534688949584961, 'test_loss': 0.2627546191215515}
    # model = MobileNetLightningModel.load_from_checkpoint("lightning_logs/version_32/checkpoints/epoch=23-step=5015.ckpt")
    # {'test_acc': 0.8241626620292664, 'test_loss': 0.28230786323547363}
    # model = MobileNetLightningModel.load_from_checkpoint("lightning_logs/version_32/checkpoints/epoch=49-step=10449.ckpt")
    # {'test_acc': 0.8355262875556946, 'test_loss': 0.3977561891078949}

    # disable randomness, dropout, etc...
    model.eval()

    datamodule = HAM10000DataModule()
    trainer = pl.Trainer( precision='bf16')
    trainer.test(model, datamodule)
