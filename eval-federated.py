import time
import os
import pytorch_lightning as pl
import torch

from torch import Tensor
from client import set_parameters_on_model

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


import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy

model = MobileNetLightningModel()
arrays =  np.load('round-1-weights.npy', allow_pickle=True)
print( arrays)
set_parameters_on_model(model.model.features, arrays[0:240])
set_parameters_on_model(model.model.classifier, arrays[240:244])
# disable randomness, dropout, etc...
model.eval()


all_image_path = glob(os.path.join('dataset', '*', '*.jpg'))
csv_file = os.path.join('dataset', 'HAM10000_metadata.csv')
df_original = pd.read_csv(csv_file)


random_index = randrange(10000)
example = df_original.iloc[[random_index]]
example = example.to_dict(orient='records')[0]
real_dx_id = example['cell_type_idx']

trans = transforms.Compose([transforms.Resize((MobileNetLightningModel.input_size,MobileNetLightningModel.input_size)),transforms.ToTensor()])
rawImage = Image.open(example['path'])
x: Tensor = trans(rawImage)
x = x.unsqueeze(0)
idToName = list(lesion_type_dict.values())
prediction = model(x)
predicted_dx_id = int(torch.argmax(prediction).item())
true_target = idToName[real_dx_id]
predicted_class = idToName[predicted_dx_id]

# Show result
print('ID: ', example['image_id'])
print('Real: ', real_dx_id)
print('Predicted: ', predicted_dx_id)
pltimage = img.imread(example['path'])
plt.imshow(pltimage)
plt.title(f'Prediction: {predicted_class} - Actual target: {true_target}')
plt.show()