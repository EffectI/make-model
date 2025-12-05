# src/utils.py
import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def compute_metrics(p):
    preds = p.predictions
    labels = p.label_ids

    unique, counts = np.unique(preds, return_counts=True)
    result_dict = dict(zip(unique, counts))
    print(f"\n[Prediction Distribution] {result_dict}")

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro') 
    
    return {
        'accuracy': acc,
        'f1': f1
    }