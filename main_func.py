import json
import argparse
import torch
from datetime import datetime
from tuning.randomsearch_tune import run_randomsearch_tune
from train.train import train_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="json config path")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)

def main():
    args = parse_args()
    tune_config = load_config(args.config)

    # load config for tuning
    data_dir = tune_config["data_dir"]
    output_path = tune_config["output_path"]
    num_trials = tune_config.get("num_trials", 512)

    # hyperparameter tuning by random search
    results_tune = run_randomsearch_tune(
        num_trials=num_trials,
        data_dir=data_dir,
        output_path=output_path,
        checkpoint_path=tune_config.get("checkpoint_path", None),
    )

    results_tune.to_csv(f"{output_path}/tune_results_df.csv", index=False)


    ### get best config from results
    if len(results_tune):
        best_config = results_tune.iloc[0].to_dict()
        best_config.pop("acc")

    else:
        with open("train_config.json") as f:
            best_config = json.load(f)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ### save the best config into a yaml file
    with open(f"./results/best_config_{timestamp}.yaml", "w") as f:
        f.write("best_config:\n")
        for key, value in best_config.items():
            f.write(f"  {key}: {value}\n")

    model_trained, acc_val = train_model(config=best_config, num_epoch=1024, data_dir=data_dir)

    #### save the trained model
    torch.save(model_trained.state_dict(), f"./results/model_{timestamp}.pth")

    print("Model trained and saved")
    print(f"Accuracy in validation set: {acc_val:.4f}")

if __name__ == "__main__":
    main()

