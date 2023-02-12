import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.consts import spots_norm_mean, spots_norm_std


class HAM10000Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df["path"][index])
        y = torch.tensor(int(self.df["cell_type_idx"][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


class HAM10000DataModule(pl.LightningDataModule):
    # units are counted from 0 to no_of_units-1
    def __init__(
        self,
        dataset_directory: str = "dataset",
        metadata_file: str = "HAM10000_metadata.csv",
        batch_size: int = 32,
        input_size: int = 224,
        unit: int = 0,
        no_of_units: int = 1,
        # for testing purposes
        minified: bool = False,
    ):
        super().__init__()

        self.dataset_directory = dataset_directory
        self.metadata_file = metadata_file
        self.batch_size = batch_size
        self.input_size = input_size
        self.unit = unit
        self.no_of_units = no_of_units
        self.minified = minified
        if minified:
            print("WARNING: minified dataset is used!")

    def setup(self, stage: str):
        print("Setting up data...")
        df = pd.read_csv(os.path.join(self.dataset_directory, self.metadata_file))

        # the code bellow can be used for making the dataset smaller for testing purposes
        if self.minified:
            df_shuffled = df.sample(frac=1, random_state=1337)
            df_split = np.array_split(df_shuffled, 20)
            df = df_split[0].reset_index()

        def get_subset(df_input: pd.DataFrame) -> pd.DataFrame:
            df_shuffled = df_input.sample(frac=1, random_state=1337)
            df_split = np.array_split(df_shuffled, self.no_of_units)
            return df_split[self.unit].reset_index()

        df_train = df[df["data_type"] == "train"]
        df_val = df[df["data_type"] == "val"]
        df_test = df[df["data_type"] == "test"]

        df_train = get_subset(df_train)
        df_val = get_subset(df_val)
        df_test = get_subset(df_test)

        print("Unit: ", self.unit)
        print("Number of units: ", self.no_of_units)
        print("Train set size: ", len(df_train))
        # print(df_train['cell_type'].value_counts())
        print("Val set size: ", len(df_val))
        # print(df_val['cell_type'].value_counts())
        print("Test set size: ", len(df_test))
        # print(df_test['cell_type'].value_counts())

        # count up the weights for each class
        # counts_of_each_value = df_train['dx'].value_counts()
        # *_, last = counts_of_each_value.items()
        # _, lowest_number_of_referents = last
        # test_list = []
        # for dx,_ in lesion_type_dict.items():
        #     data_aug_rate = lowest_number_of_referents/counts_of_each_value[dx]
        #     test_list.append(data_aug_rate)

        # print('test_list')
        # print(test_list)

        train_transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(spots_norm_mean, spots_norm_std),
            ]
        )
        # define the transformation of the val images.
        val_transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(spots_norm_mean, spots_norm_std),
            ]
        )

        self.ham_train = HAM10000Dataset(df_train, transform=train_transform)
        self.ham_val = HAM10000Dataset(df_val, transform=val_transform)
        self.ham_test = HAM10000Dataset(df_test, transform=val_transform)

    def train_dataloader(self):
        # TODO: Add more train data when doing these augmentations
        return DataLoader(self.ham_train, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.ham_val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.ham_test, batch_size=self.batch_size, num_workers=0)
