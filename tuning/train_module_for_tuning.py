import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.model import SupConModel
from model.loss_function import supcon_loss
from utiles.augmentations import augment_feature
from train.checkpoint import save_checkpoint, load_checkpoint

import os
import tempfile

from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from datetime import datetime
import ray
from ray.train import Checkpoint, report
from data.load_data import load_datasets

### train_model_tune function for ray tune

def train_model_tune(config, num_epoch=512, 
                    data_dir="/mnt/binf/eric/Mercury_Nov2024/Feature_TOO_20250506.pkl", 
                    checkpoint_path=None, identifier=None):

    #### initialize model and load hyperparameters
    model_keys = ['input_size','out1','out2', 'conv1', 'pool1', 'drop1', 'conv2', 'pool2', 'drop2', 'fc1', 'fc2', 'drop3', 'feature_dim', 'num_classes']
    config_model = {k: config[k] for k in model_keys if k in config}
    mask_prob = config['mask_prob'] if 'mask_prob' in config else 0.3
    noise = config['noise'] if 'noise' in config else 0.001
    temperature = config['temperature'] if 'temperature' in config else 0.1
    batch_size = config['batch_size'] if 'batch_size' in config else 128

    model = SupConModel(**config_model)

    dataloader_train, dataloader_test = load_datasets(data_dir = data_dir, batch_size=batch_size)

    # dataloader contains X (features), y (encoded classes), and sample_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    classification_loss = nn.CrossEntropyLoss()

    ### load checkpoint
    start_epoch = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if identifier is None:
        identifier = timestamp
    print(f"Model identifier: {identifier}")

    ### if checkpoint_path is not provided, create a new directory
    if checkpoint_path is None:
        
        checkpoint_path = f"./checkpoints_{identifier}/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
    ### if checkpoint_path is provided, check if files exist and load the latest checkpoint
    else:    
        checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.startswith("checkpoint") and f.endswith(".pth")]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_path, x)))
            start_epoch = load_checkpoint(model, optimizer, os.path.join(checkpoint_path, latest_checkpoint))
            print(f"Loaded checkpoint: {latest_checkpoint}")

    best_accuracy = 0
    early_stopping_counter = 0
    best_state = None
    for epoch in range(start_epoch, num_epoch):

        print(f"Training epoch {epoch+1}/{num_epoch}")
        ### start training
        model.train()

        total_loss = 0
        total_contrastive_loss = 0
        total_classification_loss = 0

        for i, (X, y, _) in enumerate(dataloader_train):
            X, y = X.to(device), y.to(device)

            X1 = augment_feature(X, mask_prob=mask_prob, noise=noise)
            X2 = augment_feature(X, mask_prob=mask_prob, noise=noise)

            Z1, class_pred1 = model(X1)
            Z2, class_pred2 = model(X2)

            loss_contrastive = supcon_loss(Z1, Z2, temperature)
            loss_classification = classification_loss(class_pred1, y)
            loss = loss_contrastive + loss_classification

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_contrastive_loss += loss_contrastive.item()
            total_classification_loss += loss_classification.item()
            total_loss += loss.item()

        mean_contrastive_loss = total_contrastive_loss / len(dataloader_train)
        mean_classification_loss = total_classification_loss / len(dataloader_train)
        mean_loss = total_loss / len(dataloader_train)

        print(f"Epoch {epoch+1}/{num_epoch}, Mean loss: {mean_loss:.4f}")
        print(f"Mean contrastive loss: {mean_contrastive_loss:.4f}")
        print(f"Mean classification loss: {mean_classification_loss:.4f}")
        print("=====================================================")

        ### accuracy in test set
        model.eval()
        y_test_true = []
        y_test_pred = []
        test_sample_ids = []
        with torch.no_grad():
            for i, (X_test, y_test, sample_id_test) in enumerate(dataloader_test):
                X_test = X_test.to(device)
                y_test = y_test.to(device)

                _, class_pred = model(X_test)
                y_test_true.extend(y_test.cpu().numpy())
                y_test_pred.extend(class_pred.argmax(dim=1).cpu().numpy())
                test_sample_ids.extend(sample_id_test)

        y_test_true = np.array(y_test_true)
        y_test_pred = np.array(y_test_pred)
        test_sample_ids = np.array(test_sample_ids)
        accuracy = accuracy_score(y_test_true, y_test_pred)

        print(f"Test overall accuracy: {accuracy:.4f}")
        print("=====================================================")

        # ## save checkpoints
        # if (epoch + 1) % 50 == 0:
        #     save_checkpoint(model, optimizer, epoch, f"{checkpoint_path}/checkpoint_epoch_{epoch+1}.pth")
        #     print(f"Checkpoint saved at epoch {epoch+1}")

        with tempfile.TemporaryDirectory() as tempdir:
            
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), os.path.join(tempdir, "checkpoint.pth"))
                # Send the current training result back to Tune
                ray.train.report(
                    {"best_accuracy": accuracy},
                    checkpoint=Checkpoint.from_directory(tempdir),
                )

        ### early stopping
        patience = 100
        if (epoch > 0) and (accuracy < best_accuracy):
            early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print("************************************************")
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best accuracy: {best_accuracy:.4f}")
                print("************************************************")
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                torch.save(best_state, f"./results/best_model_{identifier}.pth")
                break

        else:
            best_accuracy = accuracy
            best_state = model.state_dict()
            early_stopping_counter = 0

        # tune.report({'best_accuracy': best_accuracy})

    # return model, best_accuracy
