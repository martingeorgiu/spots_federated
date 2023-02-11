
import os
import torch

from torch import Tensor

from src.consts import lesion_type_dict
from src.get_model import getModel
from src.model import MobileNetLightningModel
 
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from src.consts import spots_norm_mean, spots_norm_std
import argparse

def main() -> None:

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-id", "--id", help="row id of the record", type=int, required=True)
    argParser.add_argument("-p", "--path", help="path of the stored model", type=str, required=True)
    argParser.add_argument("-f", "--federated", help="use federated storage format", action=argparse.BooleanOptionalAction)
    args = argParser.parse_args()

    print(f'Evaluating model: {args.path}')
    print(f'Is federated: {args.federated}')
    print(f'Row id: {args.id}')
    
    model = getModel(federated=args.federated, path=args.path)
    model.eval()
    
    csv_file = os.path.join('dataset', 'HAM10000_metadata.csv')
    df_original = pd.read_csv(csv_file)
    example = df_original.iloc[[args.id]]
    example = example.to_dict(orient='records')[0]
    real_dx_id = example['cell_type_idx']
    
    trans = transforms.Compose([transforms.Resize((MobileNetLightningModel.input_size,MobileNetLightningModel.input_size)),transforms.ToTensor(), transforms.Normalize(spots_norm_mean, spots_norm_std)])
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

if __name__ == '__main__':    
    main()