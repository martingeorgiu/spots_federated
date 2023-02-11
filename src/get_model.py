from src.model import MobileNetLightningModel
import numpy as np
from src.flower_client import set_parameters_on_model
import torch


def getModel(federated: bool, path:str) -> MobileNetLightningModel:
    if federated:
        print("Loading federated model")
        model = MobileNetLightningModel()
        arrays =  np.load(path, allow_pickle=True)
        set_parameters_on_model(model.model.features, arrays[0:240])
        set_parameters_on_model(model.model.classifier, arrays[240:244])
        return model
    else: 
        print("Loading classic model")
        return MobileNetLightningModel.load_from_checkpoint(path, map_location=torch.device('cpu'))