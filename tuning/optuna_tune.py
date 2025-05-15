import optuna
import random
import gc
import torch
import time

import pandas as pd
import numpy as np
from data.load_data import load_datasets
from train.train import train_model

def tune_trial(trial):

    # Define the hyperparameter search space
    config = {
        "input_size": 9392,
        "out1": trial.suggest_categorical("out1", [2**i for i in range(3, 7)]),       # 8, 16, 32, 64
        "out2": trial.suggest_categorical("out2", [2**i for i in range(4, 9)]),       # 16–256
        "conv1": trial.suggest_int("conv1", 1, 4),                                    # 1–4
        "pool1": trial.suggest_int("pool1", 1, 4),                                    # 1–4
        "drop1": trial.suggest_categorical("drop1", [i / 5.0 for i in range(5)]),     # 0.0–0.8
        "conv2": trial.suggest_int("conv2", 1, 4),                                    # 1–4
        "pool2": trial.suggest_int("pool2", 1, 2),                                    # 1–2
        "drop2": trial.suggest_categorical("drop2", [i / 5.0 for i in range(5)]),
        "fc1": trial.suggest_categorical("fc1", [2**i for i in range(5, 10)]),        # 32–512
        "fc2": trial.suggest_categorical("fc2", [2**i for i in range(4, 9)]),         # 16–256
        "drop3": trial.suggest_categorical("drop3", [i / 5.0 for i in range(5)]),
        "feature_dim": trial.suggest_categorical("feature_dim", [2**i for i in range(6, 10)]),  # 64–512
        "num_classes": 18,                                                           # fixed
        "mask_prob": trial.suggest_categorical("mask_prob", [i / 5.0 for i in range(6)]),       # 0.0–1.0
        "noise": 0,                                                                  # fixed
        "temperature": 0.01,                                                         # fixed
        "batch_size": trial.suggest_categorical("batch_size", [256]),                # fixed (could also just use 256)
    }


    accuracy = train_model(config=config, num_epoch=1024,
                           dataloader_train=dataloader_train,
                           dataloader_test=dataloader_test,
                           checkpoint_path="/mnt/binf/eric/Mercury_Mar2025/TOO_data/SupCon_tune/checkpoints_tmp/",
                           identifier=None)
    
    return accuracy

if __name__ == "__main__":

    data_dir="/mnt/binf/eric/Mercury_Nov2024/Feature_TOO_20250506.pkl"
    dataloader_train, dataloader_test = load_datasets(data_dir=data_dir,
                                                        batch_size=256)

    study = optuna.create_study(direction="maximize")  # or "minimize"
    study.optimize(tune_trial, n_trials=2)

    print("Best hyperparameters:", study.best_trial.params)
    print("Best accuracy:", study.best_value)

