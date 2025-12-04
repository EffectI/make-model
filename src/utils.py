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
    predictions = p.predictions
    label_ids = p.label_ids

    print(f"\n[Debug] Type: {type(predictions)}")
    if isinstance(predictions, tuple):
        print(f"[Debug] Tuple Length: {len(predictions)}")
    elif isinstance(predictions, np.ndarray):
        print(f"[Debug] Array Shape: {predictions.shape}")

    if isinstance(predictions, tuple):
        if len(predictions) > 0:
            predictions = predictions[0]
        else:
            print("Warning: 빈 튜플이 들어왔습니다. 예측값이 없습니다.")
            return {"accuracy": 0.0, "f1": 0.0}

    predictions = np.array(predictions)

    if predictions.ndim > 1:
        preds = np.argmax(predictions, axis=1)
    else:
        preds = (predictions > 0).astype(int)

    acc = accuracy_score(label_ids, preds)
    f1 = f1_score(label_ids, preds, average='macro') 
    
    return {
        'accuracy': acc,
        'f1': f1
    }