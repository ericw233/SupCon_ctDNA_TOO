import random
import gc
import torch
import time

import pandas as pd
import numpy as np
from data.load_data import load_datasets
from train.train import train_model

def sample_configs(param_space, num_samples):
    keys = list(param_space.keys())
    return [
        {k: random.choice(param_space[k]) for k in keys}
        for _ in range(num_samples)
    ]

def run_randomsearch_tune(num_trials = 1000, 
                          data_dir="/mnt/binf/eric/Mercury_Nov2024/Feature_TOO_20250506.pkl",
                          output_path="/mnt/binf/eric/Mercury_Mar2025/TOO_data/SupCon_tune/randomsearch_tune_results.csv",
                          checkpoint_path=None):

    config_space = {
    "input_size": [9392],
    "out1": [2**i for i in range(3, 7)],          # 8, 16, 32, 64
    "out2": [2**i for i in range(4, 9)],          # 16, 32, 64, 128, 256
    "conv1": [i for i in range(1, 5)],            # 1, 2, 3, 4
    "pool1": [i for i in range(1, 5)],            # 1, 2, 3, 4
    "drop1": [(i) / 5 for i in range(3)],         # 0.0, 0.2, 0.4, 0.6, 0.8
    "conv2": [i for i in range(1, 5)],            # 1, 2, 3, 4
    "pool2": [i for i in range(1, 3)],            # 1, 2
    "drop2": [(i) / 5 for i in range(3)],         # 0.0, 0.2, 0.4, 0.6, 0.8
    "fc1": [2**i for i in range(5, 10)],          # 32, 64, 128, 256, 512
    "fc2": [2**i for i in range(4, 9)],           # 16, 32, 64, 128, 256
    "drop3": [(i) / 5 for i in range(5)],         # 0.0, 0.2, 0.4, 0.6, 0.8
    "feature_dim": [2**i for i in range(6, 10)],  # 64, 128, 256, 512
    "num_classes": [18],
    "mask_prob": [(i) / 5 for i in range(4)],     # 0.0, 0.2, 0.4, 0.6
    "noise": [0.0, 0.001, 0.01],               # 0.0, 0.001, 0.01
    "temperature": [0.01, 0.1],            # 0.01, 0.1, 0.5
    "batch_size": [128],
    }
    best_acc = 0.0
    best_config = None
    configs = sample_configs(config_space, num_trials)
    results = []

    ### load dataset
    dataloader_train, dataloader_test = load_datasets(data_dir=data_dir,
                                                      batch_size=configs[0]['batch_size'])

    for i, config_i in enumerate(configs):

        print(f"============ starting trial {i+1} ============")
        print(f"Config: {config_i}")
        
        try:
            _, accuracy_i = train_model(config=config_i, num_epoch=1024, 
                                    data_dir=None,
                                    dataloader_train=dataloader_train,
                                    dataloader_test=dataloader_test,
                                    checkpoint_path=checkpoint_path,
                                    identifier=None,
                                    tuning=True)
            
            results.append({**config_i, "acc": accuracy_i})
            print("*************************************************")
            print(f"Trial {i+1} accuracy: {accuracy_i:.4f}")
            print("*************************************************")

        except Exception as e:
            print(f"Error during trial {i}: {e}")
            accuracy = 0.0

        time.sleep(15)
        torch.cuda.empty_cache()
        gc.collect()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="acc", ascending=False).reset_index(drop=True)
    results_df.to_csv(output_path, index=False)

    return results_df


