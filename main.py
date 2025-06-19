import torch
import optuna
import os
from torch.utils.data import TensorDataset, DataLoader
from model.model import PMG_model
from train.train import train_model
from train.predict import predict_dataset
from data.load_data import load_datasets
from datetime import datetime
# from tuning.ray_tune import ray_tune
from tuning.randomsearch_tune import run_randomsearch_tune
from torch.utils.tensorboard import SummaryWriter
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="/mnt/binf/eric/Mercury_Apr2025/Feature_TOO_Apr2025_resplitv50617_91.csv", help='Path to input data file')
parser.add_argument('--output_path', type=str, default="/mnt/binf/eric/Mercury_Mar2025/TOO_data/Results_PMG_ReSplitv5_0617_91_run2/", help='Path to output directory')
parser.add_argument('--identifier', type=str, default="Tune_ReSplitv5_0617_91_run2", help='Identifier for the run')
args = parser.parse_args()

data_dir = args.data_dir
output_path = args.output_path 
identifier = args.identifier

## default config
best_config = {
    "out1": 32,
    "out2": 64,
    "conv1": 4,
    "pool1": 1,
    "drop1": 0.4,
    "conv2": 2,
    "pool2": 1,
    "drop2": 0.4,
    "fc1": 128,
    "fc2": 64,
    "fc3": 128,
    "drop3": 0.4,
    "num_coarse": 8,
    "num_fine": 18,
    "feature_dim": 512,
    "mask_prob": 0.2,
    "noise": 0.0,
    "temperature": 0.1,
    "batch_size": 256,
} # best config for NJ split

if os.path.exists(output_path) is False:
    os.makedirs(output_path)
print("============== output path created =============")

################ start training #################
dataloader_train, dataloader_test, dataloader_valid = load_datasets(data_dir = data_dir, 
                                                  output_path = output_path,
                                                  batch_size = best_config['batch_size'])


input_size = dataloader_train.dataset.features.shape[2] # feature tensor of size (N, 1, input_size)
print("===================================")
print("Input size:", input_size)
best_config['input_size'] = input_size

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
### save the best config into a yaml file
with open(f"{output_path}/best_config_{timestamp}.yaml", "w") as f:
    f.write("best_config:\n")
    for key, value in best_config.items():
        f.write(f"  {key}: {value}\n")

model_hp_list = ['input_size', 'out1', 'out2', 'conv1', 'pool1', 'drop1', 'conv2', 'pool2', 'drop2',
                 'fc1', 'fc2', 'fc3', 'drop3', 'num_coarse', 'num_fine', 'feature_dim']
model_config = {key: best_config[key] for key in model_hp_list if key in best_config}
model = PMG_model(**model_config)

#### train model
# writer = SummaryWriter(log_dir=f"runs/tensorboard_{identifier}")
model_trained, best_accuracy, test_accuracy = train_model(config=best_config, num_epoch=2048,
                                           dataloader_train=dataloader_train,
                                           dataloader_test=dataloader_test,
                                           dataloader_valid=dataloader_valid,
                                           output_path=output_path,
                                           identifier=identifier,
                                           writer=None)

# writer.close()

device = next(model_trained.parameters()).device

train_score_df, _, _ = predict_dataset(model_trained, dataloader_train, device)
test_score_df, _, _ = predict_dataset(model_trained, dataloader_test, device)
valid_score_df, _, _ = predict_dataset(model_trained, dataloader_valid, device)

train_score_df.to_csv(f"{output_path}/train_score.csv", index=False)
valid_score_df.to_csv(f"{output_path}/valid_score.csv", index=False)
test_score_df.to_csv(f"{output_path}/test_score.csv", index=False)



