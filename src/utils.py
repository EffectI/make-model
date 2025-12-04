import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
    predictions = p.predictions
    labels = p.label_ids

    if isinstance(predictions, tuple):
        if len(predictions) > 0:
            predictions = predictions[0]
        else:
            print("Warning: Empty predictions tuple received. Returning 0 metrics.")
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }

    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    if len(predictions.shape) == 1:
        preds = np.array([1 if x > 0.5 else 0 for x in predictions])
    else:
        preds = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }