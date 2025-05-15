import torch
import torch.nn as nn
import os
import sys
import pandas as pd

# from model.model import DANN_1D
from tuning.train_module_for_tuning import train_model_tune

# ray tune
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from functools import partial
import ray

def ray_tune(
    num_samples=2,
    max_num_epochs=10,
    output_path="/mnt/binf/eric/Mercury_Mar2025/TOO_data/SupCon_tune/",
    data_dir="/mnt/binf/eric/Mercury_Nov2024/Feature_TOO_20250506.pkl",
):


    ray.shutdown()
    ray.init(
        address="local",
        _temp_dir="/mnt/binf/eric/ray_tmp/",
        # num_cpus=8,
        num_gpus=1,
    )

    with open(f"{output_path}/device_info_tmp.log", "w") as f:
        f.write("======================================================\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        f.write(f"GPU device: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Ray resources: {ray.available_resources()}\n")
        f.write("======================================================\n")

    config = {
        "input_size": tune.choice([9392]),
        "out1": tune.choice([2**i for i in range(3, 7)]),
        "out2": tune.choice([2**i for i in range(4, 9)]),
        "conv1": tune.choice([i for i in range(1, 5)]),
        "pool1": tune.choice([i for i in range(1, 5)]),
        "drop1": tune.choice([(i) / 5 for i in range(5)]),
        "conv2": tune.choice([i for i in range(1, 5)]),
        "pool2": tune.choice([i for i in range(1, 3)]),
        "drop2": tune.choice([(i) / 5 for i in range(5)]),
        "fc1": tune.choice([2**i for i in range(5, 10)]),
        "fc2": tune.choice([2**i for i in range(4, 9)]),
        "drop3": tune.choice([(i) / 5 for i in range(5)]),
        "feature_dim": tune.choice([2**i for i in range(6, 10)]),
        "num_classes": tune.choice([18]),
        "mask_prob": tune.choice([(i) / 5 for i in range(6)]),
        "noise": tune.choice([0.01, 0.001, 0.0001]),
        "temperature": tune.choice([0.1, 0.01, 0.001]),
        "batch_size": tune.choice([32, 64, 128]),
    }

    ### create output path
    if not os.path.exists(f"{output_path}/"):
        os.makedirs(f"{output_path}/")

    ### use tune.Tuner insead of tune.run. The latter has been deprecated
    output_path_tmp = f"{output_path}/tune_tmp/"
    if os.path.exists(output_path_tmp):
        print("Output path for ray tune already exists")
    else:
        os.makedirs(output_path_tmp, exist_ok=True)
        print("Output path for ray tune created")

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2,
    )

    tuner = tune.Tuner(

        tune.with_resources(
            trainable=partial(
                train_model_tune,
                data_dir=data_dir,
                num_epoch=1024,
            ),
            resources=train.ScalingConfig(
                # trainer_resources={"CPU": 16, "GPU": 1},
                num_workers=1,
                resources_per_worker={"GPU": 1},
                use_gpu=True,
            ),
        ),
        param_space=config,

        tune_config=tune.TuneConfig(
            metric="best_accuracy",
            mode="max",
            num_samples=num_samples,
            max_concurrent_trials=1,
            reuse_actors=True,
            # scheduler=scheduler,
        ),
        run_config=None,
    )

    results = tuner.fit()
    dfs = results.get_dataframe()
    dfs.to_csv(f"{output_path}/tune_results_df.csv", index=False)

    best_trial = results.get_best_result(metric="best_accuracy", mode="max", scope="all")

    # best_result = results.get_best_result("Valid_spec", mode="max")
    # with best_result.checkpoint.as_directory() as checkpoint_dir:
    #     state_dict = torch.load(os.path.join(checkpoint_dir, "model.pth"))
    return best_trial.config, results
