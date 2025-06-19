import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt

from data.load_data import load_datasets
from model.model import PMG_model
from model.loss_function import supcon_loss, SupConLoss
from utils.augmentations import augment_feature
from train.checkpoint import save_checkpoint, load_checkpoint
from train.predict import predict_dataset

def train_model(config, num_epoch=512,
                data_dir=None, 
                output_path="/mnt/binf/eric/Mercury_Mar2025/TOO_data/",
                dataloader_train=None, 
                dataloader_valid=None,
                dataloader_test=None,
                dataloader_ext=None,
                checkpoint_path=None, identifier=None,
                tuning=False,
                writer=None):

    # created output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # dataloader contains X (features), y (encoded classes), and sample_id
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    print(f"Using device: {device}")

    #### initialize model and load hyperparameters
    model_keys = ['input_size','out1','out2', 'conv1', 'pool1', 'drop1', 'conv2', 'pool2', 'drop2', 'fc1', 'fc2', 'fc3', 'drop3','num_coarse', 'num_fine', 'feature_dim']
    config_model = {k: config[k] for k in model_keys if k in config}
    mask_prob = config['mask_prob'] if 'mask_prob' in config else 0.3
    noise = config['noise'] if 'noise' in config else 0.001
    temperature = config['temperature'] if 'temperature' in config else 0.1
    batch_size = config['batch_size'] if 'batch_size' in config else 128

    model = PMG_model(**config_model)
    model.to(device)

    ### load data if dataloader_train and dataloader_test are not provided
    if data_dir is not None:
        dataloader_train, dataloader_test, dataloader_valid = load_datasets(data_dir = data_dir, batch_size=batch_size)

    if dataloader_valid is None and dataloader_test is not None:
        dataloader_valid = dataloader_test
        print("No validation set provided, using test set as validation set")

    ### initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) ### l2 norm
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

    classification_loss = nn.CrossEntropyLoss()
    
    ### load checkpoint
    start_epoch = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ### save the best config into a yaml file
    with open(f"./results/best_config_{timestamp}.yaml", "w") as f:
        f.write("best_config:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")

    if identifier is None:
        identifier = timestamp
    print(f"Model identifier: {identifier}")

    ### log writer for TensorBoard
    if writer is None:
        writer = SummaryWriter(log_dir=f"./runs/logs_{identifier}/")
        print(f"TensorBoard logs will be saved to: ./logs_{identifier}/")

    ### if tuning is True, ignore all checkpointing
    if not tuning:
        ### if checkpoint_path is not provided, create a new directory
        if checkpoint_path is None:
            checkpoint_path = f"./checkpoints_{identifier}/"
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
        ### if checkpoint_path is provided, check if files exist and load the latest checkpoint
        else:
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            else:
                checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.startswith("checkpoint") and f.endswith(".pth")]
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_path, x)))
                    start_epoch = load_checkpoint(model, optimizer, os.path.join(checkpoint_path, latest_checkpoint))
                    print(f"Loaded checkpoint: {latest_checkpoint}")

    ### start training
    best_accuracy = 0
    early_stopping_counter = 0
    best_state = None

    loss_contrastive_list = []
    loss_coarse_list = []
    kl_coarse_list = []
    loss_fine_list = []
    loss_fusion_list = []
    loss_list = []
    accuracy_list = []
    test_accuracy_list = []

    for epoch in range(start_epoch, num_epoch):

        print(f"Training epoch {epoch+1}/{num_epoch} (es_counter: {early_stopping_counter})")
        ### start training
        model.train()

        total_loss = 0
        total_loss_contrastive = 0
        total_loss_coarse = 0
        total_kl_coarse = 0
        total_loss_fine = 0
        total_loss_fusion = 0

        for i, (X, y_label, y_group, _) in enumerate(dataloader_train):
            X, y_label, y_group = X.to(device), y_label.to(device), y_group.to(device)

            ### augment features for contrastive learning
            X1 = augment_feature(X, mask_prob=mask_prob, noise=noise)
            X2 = augment_feature(X, mask_prob=mask_prob, noise=noise)

            out1 = model(X1, y_label)
            out2 = model(X2)

            outputs_coarse1 = out1["outputs_coarse"]
            outputs_fine1 = out1["outputs_fine"]
            outputs_fusion1 = out1["outputs_fusion"]
            outputs_projection1 = out1["outputs_projection"]
            y_coarse1 = out1["y_coarse"]

            outputs_coarse2 = out2["outputs_coarse"]
            outputs_fine2 = out2["outputs_fine"]
            outputs_fusion2 = out2["outputs_fusion"]
            outputs_projection2 = out2["outputs_projection"]
            
            ### concatenate the outputs
            output_coarse = torch.cat([outputs_coarse1, outputs_coarse2], dim=0)
            output_fine = torch.cat([outputs_fine1, outputs_fine2], dim=0)
            output_fusion = torch.cat([outputs_fusion1, outputs_fusion2], dim=0)
            # y_group = torch.cat([y_group, y_group], dim=0)
            y_coarse = torch.cat([y_coarse1, y_coarse1], dim=0)
            y_label = torch.cat([y_label, y_label], dim=0)


            ###KL divergence of coarse outputs (not included in the final loss)
            log_p_coarse1 = F.log_softmax(outputs_coarse1, dim=1)
            log_p_coarse2 = F.log_softmax(outputs_coarse2, dim=1)
            p_coarse1 = F.softmax(outputs_coarse1, dim=1)
            p_coarse2 = F.softmax(outputs_coarse2, dim=1)
            kl_coarse = F.kl_div(log_p_coarse1, p_coarse2, reduction='batchmean') + F.kl_div(log_p_coarse2, p_coarse1, reduction='batchmean')

            loss_contrastive = supcon_loss(outputs_projection1, outputs_projection2, temperature=temperature)
            # loss_coarse = classification_loss(output_coarse, y_coarse)
            loss_coarse = F.kl_div(log_p_coarse1, y_coarse1.float(), reduction='batchmean')
            loss_fine = classification_loss(output_fine, y_label)
            loss_fusion = classification_loss(output_fusion, y_label)

            loss = 0.5*loss_contrastive + 0.2*loss_coarse + 0.2*loss_fine + loss_fusion

            ### break if NaN loss occurs 
            if torch.isnan(loss_fusion):
                print(f"NaN loss at iteration {i}, config: {config}")
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_contrastive += loss_contrastive.item()
            total_loss_coarse += loss_coarse.item()
            total_kl_coarse += kl_coarse.item()
            total_loss_fine += loss_fine.item()
            total_loss_fusion += loss_fusion.item()
            total_loss += loss.item()           

        ### break if NaN loss occurs
        if torch.isnan(loss):
            print(f"NaN loss at iteration {i}, config: {config}")
            break

        mean_loss_contrastive = total_loss_contrastive / len(dataloader_train)
        mean_loss_coarse = total_loss_coarse / len(dataloader_train)
        mean_kl_coarse = total_kl_coarse / len(dataloader_train)
        mean_loss_fine = total_loss_fine / len(dataloader_train)
        mean_loss_fusion = total_loss_fusion / len(dataloader_train)
        mean_loss = total_loss / len(dataloader_train)

        print(f"Epoch {epoch+1}/{num_epoch}, Mean loss: {mean_loss:.4f}")
        print(f"Mean contrastive loss: {mean_loss_contrastive:.4f}")
        print(f"Mean coarse loss: {mean_loss_coarse:.4f}")
        print(f"Mean coarse KL: {mean_kl_coarse:.4f}")
        print(f"Mean fine loss: {mean_loss_fine:.4f}")
        print(f"Mean fusion loss: {mean_loss_fusion:.4f}")
        print(f"Mean total loss: {mean_loss:.4f}")
        print("=====================================================")
        
        loss_contrastive_list.append(mean_loss_contrastive)
        loss_coarse_list.append(mean_loss_coarse)
        kl_coarse_list.append(mean_kl_coarse)
        loss_fine_list.append(mean_loss_fine)
        loss_fusion_list.append(mean_loss_fusion)
        loss_list.append(mean_loss)

        ############################################
        ### accuracy in test set
        model.eval()
        # y_test_true = []
        # y_test_pred = []
        # test_sample_ids = []
        # with torch.no_grad():
        #     for i, (X_test, y_label_test, _, sample_id_test) in enumerate(dataloader_test):
        #         X_test = X_test.to(device)
        #         y_label_test = y_label_test.to(device)

        #         output_test = model(X_test)
        #         fusion_test = output_test["outputs_fusion"]

        #         y_test_true.extend(y_label_test.cpu().numpy())
        #         y_test_pred.extend(fusion_test.argmax(dim=1).cpu().numpy())
        #         test_sample_ids.extend(sample_id_test)

        # y_test_true = np.array(y_test_true)
        # y_test_pred = np.array(y_test_pred)
        # test_sample_ids = np.array(test_sample_ids)
        # accuracy = accuracy_score(y_test_true, y_test_pred)

        # accuracy_list.append(accuracy)

        ### predict on validation set
        _, accuracy, accuracy2 = predict_dataset(model, dataloader_valid, device)
        accuracy_list.append(accuracy)

        print(f"Valid top1 accuracy: {accuracy:.4f} ({best_accuracy:.4f})")
        print(f"Valid top2 accuracy: {accuracy2:.4f}")
        print("*******************************************************")

        ### predict on test set
        _, test_acc1, test_acc2 = predict_dataset(model, dataloader_test, device)
        test_accuracy_list.append(test_acc1)
        print(f"Test top1 accuracy: {test_acc1:.4f}")
        print(f"Test top2 accuracy: {test_acc2:.4f}")
        print("=====================================================")

        ### external test results
        if dataloader_ext is not None:
            _, ext_acc1, ext_acc2 = predict_dataset(model, dataloader_ext, device)
            print(f"External test top1 accuracy: {ext_acc1:.4f}")
            print(f"External test top2 accuracy: {ext_acc2:.4f}")
            print("=====================================================")

        ### add logs to TensorBoard if writer is provided
        if writer is not None:
            writer.add_scalar('Loss/contrastive', mean_loss_contrastive, epoch)
            writer.add_scalar('Loss/coarse', mean_loss_coarse, epoch)
            writer.add_scalar('Loss/coarse_kl', mean_kl_coarse, epoch)
            writer.add_scalar('Loss/fine', mean_loss_fine, epoch)
            writer.add_scalar('Loss/fusion', mean_loss_fusion, epoch)
            writer.add_scalar('Loss/total', mean_loss, epoch)
            writer.add_scalar('Accuracy/top1', accuracy, epoch)
            writer.add_scalar('Accuracy/top2', accuracy2, epoch)
            writer.add_scalar('Test_Accuracy/top1', test_acc1, epoch)
            writer.add_scalar('Test_Accuracy/top2', test_acc2, epoch)
            if dataloader_ext is not None:
                writer.add_scalar('External_Accuracy/top1', ext_acc1, epoch)
                writer.add_scalar('External_Accuracy/top2', ext_acc2, epoch)

        ### save checkpoints
        if not tuning:
            if (epoch + 1) % 200 == 0:
                save_checkpoint(model, optimizer, epoch, f"{checkpoint_path}/checkpoint_epoch_{epoch+1}.pth")
                print(f"Checkpoint saved at epoch {epoch+1}")
        
        ### early stopping
        patience = 150
        if (epoch > 0) and (accuracy <= best_accuracy):
            early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print("************************************************")
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best accuracy: {best_accuracy:.4f}")
                print("************************************************")
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if not tuning:
                    torch.save(best_state, f"{output_path}/best_model_{identifier}_earlystopping.pth")
                
                break
        else:
            best_accuracy = accuracy
            best_state = deepcopy(model.state_dict())
            early_stopping_counter = 0

    ### save the final model and results
    if not tuning:
        torch.save(best_state, f"{output_path}/best_model_{identifier}.pth")
        print(f"Best model saved at {output_path}/best_model_{identifier}.pth")

    model.load_state_dict(best_state)
    model.eval()
    _, acc1, acc2 = predict_dataset(model, dataloader_valid, device)
    print(f"Reload valid accuracy: {acc1:.4f}, {acc2:.4f}")

    _, test_acc1, test_acc2 = predict_dataset(model, dataloader_test, device)
    print(f"Reload testset accuracy: {test_acc1:.4f}, {test_acc2:.4f}")

    if dataloader_ext is not None:
        _, ext_acc1, ext_acc2 = predict_dataset(model, dataloader_ext, device)
        print(f"Reload external testset accuracy: {ext_acc1:.4f}, {ext_acc2:.4f}")

    ### plot loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label='Loss')
    plt.plot(loss_contrastive_list, label='Contrastive Loss')
    plt.plot(loss_coarse_list, label='Coarse Loss')
    plt.plot(kl_coarse_list, label='Coarse KL Loss')
    plt.plot(loss_fine_list, label='Fine Loss')
    plt.plot(loss_fusion_list, label='Fusion Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_list, label='Valid Accuracy')
    plt.plot(test_accuracy_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.savefig(f"{output_path}/loss_accuracy_{identifier}.png")
    plt.show()

    return model, best_accuracy, test_acc1
