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
        # norm_mean = [0.7630392, 0.5456477, 0.57004845]
        # norm_std = [0.1409286, 0.15261266, 0.16997074]
        # df = df_og.copy()

        # df_undup = df.groupby('lesion_id').count()
        # # now we filter out lesion_id's that have only one image associated with it
        # df_undup = df_undup[df_undup['image_id'] == 1]
        # df_undup.reset_index(inplace=True)

        # def get_duplicates(x):
        #     unique_list = list(df_undup['lesion_id'])
        #     if x in unique_list:
        #         return 'unduplicated'
        #     else:
        #         return 'duplicated'

        # # create a new colum that is a copy of the lesion_id column
        # df['duplicates'] = df['lesion_id']
        # # apply the function to this new column
        # df['duplicates'] = df['duplicates'].apply(get_duplicates)
        # df_undup = df[df['duplicates'] == 'unduplicated']
        # y = df_undup['cell_type_idx']
        # _, df_val = train_test_split(df_undup, test_size=0.3, random_state=1337, stratify=y)

        # # This set will be df_original excluding all rows that are in the val set
        # # This function identifies if an image is part of the train or val set.
        # def get_val_rows(x):
        #     # create a list of all the lesion_id's in the val set
        #     val_list = list(df_val['image_id'])
        #     if str(x) in val_list:
        #         return 'val'
        #     else:
        #         return 'train'

        # # identify train and val rows
        # # create a new colum that is a copy of the image_id column
        # df['train_or_val'] = df['image_id']
        # # apply the function to this new column
        # df['train_or_val'] = df['train_or_val'].apply(get_val_rows)
        # # filter out train rows
        # df_train = df[df['train_or_val'] == 'train']
        

        # print('df_og')
        # print(df_og.count())
      
        universal_transform = transforms.Compose([transforms.Resize((self.input_size,self.input_size)), transforms.ToTensor()])

        df = df.drop_duplicates(subset=['lesion_id'], keep='first')
        df = df.reset_index()
        print('df')
        print(len(df.index))
        print(df['cell_type'].value_counts())

        y = df['cell_type_idx']
        df_train, df_val = train_test_split(df, test_size=0.2, random_state=1337,stratify=y)


        df_train= df_train.reset_index()
        df_val =df_val.reset_index()
        print('df_train')
        print(len(df_train))
        print(df_train['path'])
        print(df_train['path'][2])
        print(len(df_train.index))
        print(df_train['cell_type'].value_counts())
        print('df_val')
        print(len(df_val.index))
        print(df_val['cell_type'].value_counts())


        # counts_of_each_value = df_train['dx'].value_counts()
        # highest_type, highest_number_of_referents = next(x for x in counts_of_each_value.items())
        # test_dict = {}
        # test_list = []
        # for dx,_ in lesion_type_dict.items():
        #     data_aug_rate = 1
        #     if(dx != highest_type):
        #         data_aug_rate = floor(highest_number_of_referents/counts_of_each_value[dx])
        #     test_list.append(data_aug_rate)
        #     # test_dict[lesion_type_id.index(dx)] = data_aug_rate
        
        # print('test_list')
        # print(test_list)

        self.ham_train = HAM10000Dataset(df_train,transform=universal_transform)
        self.ham_val = HAM10000Dataset(df_val,transform=universal_transform)
        self.ham_test = HAM10000Dataset(df,transform=universal_transform)



    def train_dataloader(self):
        # TODO: Add more train data when doing these augmentations
        return DataLoader(self.ham_train, batch_size=self.batch_size, num_workers=10)

    def val_dataloader(self):
        # val_transform = transforms.Compose([transforms.Resize((self.input_size,self.input_size)), transforms.ToTensor()])
        return DataLoader(self.ham_val, batch_size=self.batch_size, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.ham_test, batch_size=self.batch_size, num_workers=10)


