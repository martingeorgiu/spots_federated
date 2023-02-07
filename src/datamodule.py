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
        df_og = pd.read_csv(os.path.join(self.dataset_directory, self.metadata_file))
        # path = glob(os.path.join(self.dataset_directory, '*', '*.jpg'))
        # norm_mean,norm_std = compute_img_mean_std(path)
        norm_mean = [0.7630392, 0.5456477, 0.57004845]
        norm_std = [0.1409286, 0.15261266, 0.16997074]
        df = df_og.copy()

        df_undup = df.groupby('lesion_id').count()
        # now we filter out lesion_id's that have only one image associated with it
        df_undup = df_undup[df_undup['image_id'] == 1]
        df_undup.reset_index(inplace=True)

        def get_duplicates(x):
            unique_list = list(df_undup['lesion_id'])
            if x in unique_list:
                return 'unduplicated'
            else:
                return 'duplicated'

        # create a new colum that is a copy of the lesion_id column
        df['duplicates'] = df['lesion_id']
        # apply the function to this new column
        df['duplicates'] = df['duplicates'].apply(get_duplicates)
        df_undup = df[df['duplicates'] == 'unduplicated']
        y = df_undup['cell_type_idx']
        _, df_val = train_test_split(df_undup, test_size=0.2, random_state=1337, stratify=y)

        # This set will be df_original excluding all rows that are in the val set
        # This function identifies if an image is part of the train or val set.
        def get_val_rows(x):
            # create a list of all the lesion_id's in the val set
            val_list = list(df_val['image_id'])
            if str(x) in val_list:
                return 'val'
            else:
                return 'train'

        # identify train and val rows
        # create a new colum that is a copy of the image_id column
        df['train_or_val'] = df['image_id']
        # apply the function to this new column
        df['train_or_val'] = df['train_or_val'].apply(get_val_rows)
        # filter out train rows
        df_train = df[df['train_or_val'] == 'train']
        
        y = df_train['cell_type_idx']
        df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=1337, stratify=y)
        df_train = df_train.reset_index()
        df_val = df_val.reset_index()

        print("Train set size: ", len(df_train))
        print("Val set size: ", len(df_val))
        print("Test set size: ", len(df_test))


        train_transform = transforms.Compose([transforms.Resize((self.input_size,self.input_size)),transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                                transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
        # define the transformation of the val images.
        val_transform = transforms.Compose([transforms.Resize((self.input_size,self.input_size)), transforms.ToTensor(),
                                            transforms.Normalize(norm_mean, norm_std)])

        self.ham_train = HAM10000Dataset(df_train,transform=train_transform)
        self.ham_val = HAM10000Dataset(df_val,transform=val_transform)
        self.ham_test = HAM10000Dataset(df_test,transform=val_transform)



    def train_dataloader(self):
        # TODO: Add more train data when doing these augmentations
        return DataLoader(self.ham_train, batch_size=self.batch_size, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.ham_val, batch_size=self.batch_size, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.ham_test, batch_size=self.batch_size, num_workers=10)


