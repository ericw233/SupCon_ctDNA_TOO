import torch
import optuna
from torch.utils.data import TensorDataset, DataLoader
from model.model import SupConModel
from train.train import train_model
from data.load_data import load_datasets
from datetime import datetime
from tuning.ray_tune import ray_tune
from tuning.randomsearch_tune import run_randomsearch_tune

### ray tune
# best_config, tune_results = ray_tune(
    
#     num_samples=2,
#     max_num_epochs=50,
#     output_path="/mnt/binf/eric/Mercury_Mar2025/TOO_data/SupCon_tune/",
#     data_dir="/mnt/binf/eric/Mercury_Nov2024/Feature_TOO_20250506.pkl",

# )

### random search tune

# results_tune = run_randomsearch_tune(
#     num_trials=256,
#     data_dir="/mnt/binf/eric/Mercury_Nov2024/Feature_TOO_20250506.pkl",
#     output_path="/mnt/binf/eric/Mercury_Mar2025/TOO_data/SupCon_tune/randomsearch_tune_results_0509.csv",
# )

results_tune_1024 = run_randomsearch_tune(
    num_trials=4096,
    data_dir="/mnt/binf/eric/Mercury_Nov2024/Feature_TOO_20250506.pkl",
    output_path="/mnt/binf/eric/Mercury_Mar2025/TOO_data/SupCon_tune/randomsearch_tune_results_0514.csv",
    checkpoint_path="/mnt/binf/eric/Mercury_Mar2025/TOO_data/SupCon_tune/checkpoints_tmp/",
)



### default config
# best_config = {
#     "out1": 16,
#     "out2": 128,
#     "conv1": 2,
#     "pool1": 2,
#     "drop1": 0.0,
#     "conv2": 4,
#     "pool2": 1,
#     "drop2": 0.4,
#     "fc1": 128,
#     "fc2": 16,
#     "drop3": 0.2,
#     "feature_dim": 64,
#     "num_classes": 18,
#     "mask_prob": 0.3,
#     "noise": 0.001,
#     "temperature": 0.1,
#     "batch_size": 128,
# }

# ### load data
# data_dir = "/mnt/binf/eric/Mercury_Nov2024/Feature_TOO_20250506.pkl"
# dataloader_train, dataloader_test = load_datasets(data_dir)

# input_size = dataloader_train.dataset.features.shape[2] # feature tensor of size (N, 1, input_size)
# best_config['input_size'] = input_size

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# ### save the best config into a yaml file
# with open(f"./results/best_config_{timestamp}.yaml", "w") as f:
#     f.write("best_config:\n")
#     for key, value in best_config.items():
#         f.write(f"  {key}: {value}\n")


# model = SupConModel(**best_config)
# best_config['mask_prob'] = 0.3
# best_config['noise'] = 0.001
# model_trained, best_accuracy = train_model(config, dataloader_train, dataloader_test, num_epoch=1024, 
#                             temperature=0.1, 
#                             mask_prob=best_config['mask_prob'], 
#                             noise=best_config['noise'])


