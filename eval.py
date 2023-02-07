import time
import os
import pytorch_lightning as pl
import torch

from torch import Tensor

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

model = MobileNetLightningModel.load_from_checkpoint("lightning_logs/version_27/checkpoints/epoch=0-step=208.ckpt")
# disable randomness, dropout, etc...
model.eval()


all_image_path = glob(os.path.join('dataset', '*', '*.jpg'))
csv_file = os.path.join('dataset', 'HAM10000_metadata.csv')
df_original = pd.read_csv(csv_file)


random_index = randrange(10000)
example = df_original.iloc[[10000]]
example = example.to_dict(orient='records')[0]
real_dx_id = example['cell_type_idx']

norm_mean = [0.7630392, 0.5456477, 0.57004845]
norm_std = [0.1409286, 0.15261266, 0.16997074]
trans = transforms.Compose([transforms.Resize((MobileNetLightningModel.input_size,MobileNetLightningModel.input_size)),transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
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