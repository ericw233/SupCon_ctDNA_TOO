import torch
from torch.utils.data import DataLoader, Dataset
from data.preprocess_data import make_preprocessor
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

class MyDataset(Dataset):
    def __init__(self, features, labels, sampleids):
        self.features = features
        self.labels = labels
        self.sampleids = sampleids

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.sampleids[index]

    def __len__(self):
        return len(self.sampleids)


def load_datasets(data_dir="/mnt/binf/eric/Mercury_Nov2024/Feature_TOO_20250506.pkl",
                  batch_size=128):

    ### load data and encode labels
    if data_dir.endswith(".pkl"):
        data_df = pd.read_pickle(data_dir)
    elif data_dir.endswith(".csv"):
        data_df = pd.read_csv(data_dir)

    ### identify numeric columns
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()

    data_train = data_df.loc[data_df['train'] == "train",:]
    data_test = data_df.loc[data_df['train'] != "train",:]

    labelencoder = LabelEncoder()
    data_train.loc[:,'Label_encoded'] = labelencoder.fit_transform(data_train['GroupLevel2'])
    data_test.loc[:,'Label_encoded'] = labelencoder.transform(data_test['GroupLevel2'])

    ### preprocess features
    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(data_train.loc[:,numeric_cols])
    X_test = preprocessor.transform(data_test.loc[:,numeric_cols])

    ### create dataloaders

    dataloader_train = DataLoader(
        MyDataset(
            features=torch.tensor(X_train, dtype=torch.float).unsqueeze(1),
            labels=torch.tensor(data_train['Label_encoded'].values, dtype=torch.long),
            sampleids=data_train['SampleID'].values
        ),
        batch_size=128,
        shuffle=True
    )

    dataloader_test = DataLoader(
        MyDataset(
            features=torch.tensor(X_test, dtype=torch.float).unsqueeze(1),
            labels=torch.tensor(data_test['Label_encoded'].values, dtype=torch.long),
            sampleids=data_test['SampleID'].values
        ),
        batch_size=128,
        shuffle=False
    )

    return dataloader_train, dataloader_test



