import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def predict_dataset(model, dataloader, device):

    model.eval()
    with torch.no_grad():

        ### evaluate on test set
        y_test_true = []
        y_test_1st = []
        y_test_2nd = []
        test_sample_ids = []
        score_test = None

        for i, (X_test, y_label_test, _, sample_id_test) in enumerate(dataloader):
            X_test = X_test.to(device)
            y_label_test = y_label_test.to(device)

            output_test = model(X_test)
            fusion_test = output_test["outputs_fusion"]

            y_test_true.extend(y_label_test.cpu().numpy())
            y_test_1st.extend(fusion_test.argmax(dim=1).cpu().numpy())
            y_test_2nd.extend(fusion_test.argsort(dim=1)[:, -2].cpu().numpy())

            test_sample_ids.extend(sample_id_test)
            score_test = fusion_test.cpu().numpy() if score_test is None else np.concatenate((score_test, fusion_test.cpu().numpy()), axis=0)

        ### process and save test results
        score_test = pd.DataFrame(score_test, columns=[f'Score_{i}' for i in range(score_test.shape[1])])
        test_results_df = pd.DataFrame({
            'SampleID': test_sample_ids,
            'Label': y_test_true,
            'Pred_too1': y_test_1st,
            "Pred_2nd": y_test_2nd
        })
        
        test_results_df['Pred_too2'] = np.where(
            test_results_df['Pred_2nd'] == test_results_df['Label'], 
            test_results_df['Pred_2nd'], 
            test_results_df['Pred_too1']
        )
        test_results_df = pd.concat([test_results_df, score_test], axis=1)

        ### calculate accuracy
        test_acc_1st = accuracy_score(y_test_true, test_results_df['Pred_too1'])
        test_acc_2nd = accuracy_score(y_test_true, test_results_df['Pred_too2'])
        print(f"Accuracy TOO1: {test_acc_1st:.4f}, TOO2: {test_acc_2nd:.4f}")

    return test_results_df, test_acc_1st, test_acc_2nd

