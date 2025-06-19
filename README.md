# SupCon_TOO

SupCon_TOO is a PyTorch-based deep learning model that combines supervised contrastive learning and hierarchical classification algorithms for Tissue-Of-Origin (TOO) analysis. The repository includes modules for data preprocessing, model training, and evaluation.

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
- [Configuration](#configuration)
- [Key Components](#key-components)
  - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Loss Functions](#loss-functions)
  - [Training and Evaluation](#training-and-evaluation)
- [Model Structure Diagram](#model-structure-diagram)
- [Results](#results)
- [License](#license)

---

## Features

- **Supervised Contrastive Learning**: Implements supervised contrastive loss for feature learning
- **Hierarchical Classification**: A hierarchical model with coarse and fine classification heads
- **Data Augmentation**: Includes feature masking and noise injection for robust training
- **Checkpointing**: Save and load model checkpoints for resuming training
- **TensorBoard Integration**: Logs training metrics for visualization

---

## Repository Structure

```
SupCon_TOO/
├── LICENSE
├── README.md
├── main.py                    # Main training script
├── main_func.py              # Functional training interface
├── train_config.json         # Training configuration
├── tune_config.json          # Tuning configuration
├── data/
│   ├── load_data.py          # Data loading utilities
│   └── preprocess_data.py    # Data preprocessing functions
├── model/
│   ├── model.py              # PMG model implementation
│   └── loss_function.py      # Supervised contrastive loss
├── train/
│   ├── train.py              # Training logic
│   ├── predict.py            # Model evaluation
│   └── checkpoint.py         # Model checkpointing
├── tuning/
│   ├── optuna_tune.py        # Optuna hyperparameter tuning
│   ├── randomsearch_tune.py  # Random search tuning
│   ├── ray_tune.py           # Ray Tune integration
│   └── train_module_for_tuning.py
├── utils/
│   └── augmentations.py      # Data augmentation utilities
├── results/                  # Output directory for results
└── runs/                     # TensorBoard logs
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/SupCon_TOO.git
   cd SupCon_TOO
   ```

2. Install dependencies:
   ```bash
   conda env create -f env.yml
   conda activate SupCon_TOO
   ```

---

## Usage

### Training

To train the model using the main script:

```bash
python main.py --data_dir <path_to_data> --output_path <path_to_output> --identifier <run_identifier>
```

**Example:**
```bash
python main.py \
  --data_dir /path/to/your/dataset.csv \
  --output_path ./results/too_exp_1/ \
  --identifier "too_exp_1"
```

**Command Line Arguments:**
- `--data_dir`: Path to input CSV data file
- `--output_path`: Directory to save results
- `--identifier`: Unique identifier for the training run

---

## Configuration

The model uses a configuration dictionary with the following key parameters:

```python
best_config = {
    "out1": 32,           # First conv output channels
    "out2": 64,           # Second conv output channels
    "conv1": 4,           # First conv kernel size
    "pool1": 1,           # First pooling kernel size
    "drop1": 0.4,         # First dropout rate
    "conv2": 2,           # Second conv kernel size
    "pool2": 1,           # Second pooling kernel size
    "drop2": 0.4,         # Second dropout rate
    "fc1": 128,           # First FC layer size
    "fc2": 64,            # Second FC layer size
    "fc3": 128,           # Third FC layer size
    "drop3": 0.4,         # Third dropout rate
    "num_coarse": 8,      # Number of coarse classes
    "num_fine": 18,       # Number of fine classes
    "feature_dim": 512,   # Feature dimension for contrastive learning
    "mask_prob": 0.2,     # Probability for feature masking
    "noise": 0.0,         # Noise level for augmentation
    "temperature": 0.1,   # Temperature for contrastive loss
    "batch_size": 256,    # Training batch size
}
```

---

## Key Components

### Data Loading and Preprocessing

- **File**: `data/load_data.py`
- **Description**: Loads datasets from a .csv/.pickle file, performs train/test/validation splits, applies preprocessing pipelines, and creates PyTorch DataLoaders

### Model Architecture

- **File**: `model/model.py`
- **Description**: Implements the supervised contrastive learning and hierarchical classification model featuring:
  - Convolutional feature extraction layers
  - Coarse and fine classification heads
  - Fusion classifier for final predictions
  - Projection head for contrastive learning

### SupCon Loss Functions

- **File**: `model/loss_function.py`
- **Description**: Implements supervised contrastive loss for learning discriminative feature representations

### Training and Evaluation

- **File**: `train/train.py`
- **Description**: Handles complete training pipeline including:
  - Model training with early stopping
  - Validation monitoring
  - Model checkpointing
  - TensorBoard logging

- **File**: `train/predict.py`
- **Description**: Evaluates trained models on datasets and generates prediction scores and accuracy metrics

---

## Model Structure Diagram

```
Input Features (N, 1, input_size)
      ↓
┌─────────────────────┐
│  Convolutional      │
│  Feature Extraction │
│  - Conv1D + Pool    │
│  - Dropout          │
└─────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│               Branch Split              │
├─────────────────┬───────────────────────┤
│  Coarse Head    │    Fine Head          │
│  (num_coarse)   │    (num_fine)         │
└─────────────────┴───────────────────────┘
      ↓                      ↓
┌─────────────────────────────────────────┐
│          Fusion Classifier              │
│     (Combines coarse + fine)            │
└─────────────────────────────────────────┘
      ↓
Final Predictions

Regularizing Components:
┌─────────────────────┐
│  Projection Head    │
│  (for contrastive   │
│   learning)         │
└─────────────────────┘
```

The PMG model uses a hierarchical approach where:
1. **Feature Extraction**: Convolutional layers extract features from input data
2. **Multi-Granularity Classification**: Separate heads for coarse and fine-grained predictions
3. **Fusion**: Combines coarse and fine predictions for final output
4. **Contrastive Learning**: Projection head enables supervised contrastive learning

---

## Results

The framework generates comprehensive outputs:

- **Model Checkpoints**: Saved in `output_path/` with timestamp
- **Configuration Files**: YAML files with hyperparameters
- **Prediction Scores**: CSV files for train/validation/test sets
- **TensorBoard Logs**: Training metrics and visualizations in `runs/`

**Output Files:**
- `train_score.csv`
- `valid_score.csv`
- `test_score.csv`

---
