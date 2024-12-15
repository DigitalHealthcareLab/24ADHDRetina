import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import log_loss, brier_score_loss


def calculate_scores(preds, labels, threshold = 0.5) : 
    if preds.ndim == 2 : 
        preds = preds[:, 1]
    hard_preds = np.where(preds > threshold, 1, 0)
    
    auroc = roc_auc_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, hard_preds).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity + 1e-8)
    nll = log_loss(labels, preds)
    brier = brier_score_loss(labels, preds)
    scores = {
        'AUROC' : auroc,
        'Sensitivity' : sensitivity,
        'Specificity' : specificity,
        'F1-Score' : f1,
        'PPV' : ppv,
        'NPV' : npv,
        'NLL' : nll,
        'Brirer' : brier,
    }
    return scores

def calculate_youden_index(preds, labels) : 
    if preds.ndim == 2 : 
        preds = preds[:, 1]
        
    youden_values = []
    for threshold in np.arange(0, 1.01, 0.01) : 
        hard_pred = np.where(preds > threshold, 1, 0)
        tn, fp, fn, tp = confusion_matrix(labels, hard_pred).ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        youden_v = sensitivity + specificity - 1
        youden_values.append([threshold, youden_v])
    youden_df = pd.DataFrame(youden_values, columns = ['threshold', 'youden_index'])
    max_v = youden_df['youden_index'].max()
    youden_df = youden_df.query(f"youden_index == {max_v}")
    threshold = np.median(youden_df['threshold'].values)
    return threshold
        