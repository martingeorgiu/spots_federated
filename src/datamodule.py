from glob import glob
from math import floor
import os
import torch
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import random_split
import pytorch_lightning as pl
from PIL import Image
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

from src.utils import compute_img_mean_std

lesion_type_dict = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions ',
    'df': 'Dermatofibroma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions',
    'mel': 'Melanoma',
}

lesion_type_id = [
    'akiec',
    'bcc',
    'bkl',
    'df',
    'nv',
    'vasc',
    'mel',
]

class HAM10000Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y

class HAM10000DataModule(pl.LightningDataModule):
    def __init__(self, dataset_directory: str = "dataset", metadata_file: str = "HAM10000_metadata.csv", batch_size: int = 32, input_size: int = 224):
        super().__init__()

        self.dataset_directory = dataset_directory
        self.metadata_file = metadata_file
        self.batch_size = batch_size
        self.input_size = input_size

    def setup(self, stage: str):
        print("Setting up data...")   
        df = pd.read_csv(os.path.join(self.dataset_directory, self.metadata_file))
        # path = glob(os.path.join(self.dataset_directory, '*', '*.jpg'))
        # norm_mean,norm_std = compute_img_mean_std(path)

        allimages_norm_mean = [0.7630392, 0.5456477, 0.57004845]
        allimages_norm_std = [0.1409286, 0.15261266, 0.16997074]
    

        df_train = df[df['data_type'] == 'train']
        df_val = df[df['data_type'] == 'val']
        df_test = df[df['data_type'] == 'test']

        df_train = df_train.reset_index()
        df_val = df_val.reset_index()
        df_test = df_test.reset_index()

        print("Train set size: ", len(df_train))
        print(df_train['cell_type'].value_counts())
        print("Val set size: ", len(df_val))
        print(df_val['cell_type'].value_counts())
        print("Test set size: ", len(df_test))
        print(df_test['cell_type'].value_counts())

        counts_of_each_value = df_train['dx'].value_counts()
        *_, last = counts_of_each_value.items()
        _, lowest_number_of_referents = last
        test_list = []
        for dx,_ in lesion_type_dict.items():
            data_aug_rate = lowest_number_of_referents/counts_of_each_value[dx]
            test_list.append(data_aug_rate)
        
        print('test_list')
        print(test_list)

        train_transform = transforms.Compose([transforms.Resize((self.input_size,self.input_size)),transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                                transforms.ToTensor(), transforms.Normalize(allimages_norm_mean, allimages_norm_std)])
        # define the transformation of the val images.
        val_transform = transforms.Compose([transforms.Resize((self.input_size,self.input_size)), transforms.ToTensor(),
                                            transforms.Normalize(allimages_norm_mean, allimages_norm_std)])

        self.ham_train = HAM10000Dataset(df_train,transform=train_transform)
        self.ham_val = HAM10000Dataset(df_val,transform=val_transform)
        self.ham_test = HAM10000Dataset(df_test,transform=val_transform)



    def train_dataloader(self):
        # TODO: Add more train data when doing these augmentations
        return DataLoader(self.ham_train, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.ham_val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.ham_test, batch_size=self.batch_size, num_workers=0)


