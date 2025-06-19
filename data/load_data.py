import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from data.preprocess_data import make_preprocessor
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import joblib

class MyDataset(Dataset):
    def __init__(self, features, labels, groups, sampleids):
        self.features = features
        self.labels = labels
        self.groups = groups
        self.sampleids = sampleids

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.groups[index], self.sampleids[index]

    def __len__(self):
        return len(self.sampleids)


def load_datasets(data_dir="/mnt/binf/eric/Mercury_Nov2024/Feature_TOO_20250506_encoded.pkl",
                  output_path="/mnt/binf/eric/Mercury_Mar2025/TOO_data/",
                  batch_size=128):

    ### load data and encode labels
    if data_dir.endswith(".pkl"):
        data_df = pd.read_pickle(data_dir)
    elif data_dir.endswith(".csv"):
        data_df = pd.read_csv(data_dir)

    ### identify numeric columns
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['Label_encoded', 'Group_encoded', 'SampleID', 'train']]

    data_train = data_df.loc[data_df['train'] == "train",:]
    data_test = data_df.loc[data_df['train'] == "test",:]
    data_valid = data_df.loc[data_df['train'] == "valid",:]

    ### encoded labels are included in the data_df
    encoder_df = data_df.loc[:,["GroupLevel2", "Label_encoded"]].drop_duplicates()
    encoder_dict = dict(zip(encoder_df['Label_encoded'], encoder_df['GroupLevel2']))
    joblib.dump(encoder_dict, output_path + 'LabelEncoder.joblib')

    # labelencoder = LabelEncoder()
    # data_train.loc[:,'Label_encoded'] = labelencoder.fit_transform(data_train['GroupLevel2'])
    # data_test.loc[:,'Label_encoded'] = labelencoder.transform(data_test['GroupLevel2'])

    # groupencoder = LabelEncoder()
    # data_train.loc[:,'Group_encoded'] = groupencoder.fit_transform(data_train['GroupLevel2_mapped'])
    # data_test.loc[:,'Group_encoded'] = groupencoder.transform(data_test['GroupLevel2_mapped'])

    ### preprocess features
    preprocessor = make_preprocessor()
    X_train = preprocessor.fit_transform(data_train.loc[:,numeric_cols])
    X_test = preprocessor.transform(data_test.loc[:,numeric_cols])
    X_valid = preprocessor.transform(data_valid.loc[:,numeric_cols])

    y_train = data_train['Label_encoded'].values # y_train is needed for WeightedRandomSampler

    ### save preprocessor
    joblib.dump(preprocessor, output_path + 'preprocessor.joblib')
    
    ### create dataloaders
    ###### weighted sampler for train set
    
    class_counts = torch.bincount(torch.tensor(y_train, dtype=torch.long))
    class_weights = 1.0 / class_counts.float() 
    mask = class_weights > class_weights.quantile(0.75)
    class_weights_adj = class_weights.clone()
    class_weights_adj[mask] *= 0.25

    weights_list = class_weights[y_train]
    weights_list_adj = class_weights_adj[y_train]

    weightedSampler = WeightedRandomSampler(weights=weights_list, num_samples=batch_size, replacement=True)

    dataloader_train = DataLoader(
        MyDataset(
            features=torch.tensor(X_train, dtype=torch.float).unsqueeze(1),
            labels=torch.tensor(y_train, dtype=torch.long),
            groups=torch.tensor(data_train['Group_encoded'].values, dtype=torch.long),
            sampleids=data_train['SampleID'].values
        ),
        batch_size=batch_size,
        sampler=weightedSampler,
        # shuffle=True
    )

    dataloader_test = DataLoader(
        MyDataset(
            features=torch.tensor(X_test, dtype=torch.float).unsqueeze(1),
            labels=torch.tensor(data_test['Label_encoded'].values, dtype=torch.long),
            groups=torch.tensor(data_test['Group_encoded'].values, dtype=torch.long),
            sampleids=data_test['SampleID'].values
        ),
        
        batch_size=batch_size,
        shuffle=False
    )

    if X_valid.shape[0] > 0:
        dataloader_valid = DataLoader(
            MyDataset(
                features=torch.tensor(X_valid, dtype=torch.float).unsqueeze(1),
                labels=torch.tensor(data_valid['Label_encoded'].values, dtype=torch.long),
                groups=torch.tensor(data_valid['Group_encoded'].values, dtype=torch.long),
                sampleids=data_valid['SampleID'].values
            ),
            batch_size=batch_size,
            shuffle=False
        )
        
        return dataloader_train, dataloader_test, dataloader_valid
    
    else:
        return dataloader_train, dataloader_test, None



