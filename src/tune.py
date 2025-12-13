import os
import yaml
import torch
import numpy as np
import optuna
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_loader import get_tokenized_datasets

# ---------------------------------------------------------
# 1. 설정 및 데이터 로드
# ---------------------------------------------------------
CONFIG_PATH = "make-model/config/koelectra.yaml" 
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    base_cfg = yaml.safe_load(f)

MODEL_NAME = base_cfg['model']['name']
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("데이터셋 로드 중...")
train_dataset, eval_dataset = get_tokenized_datasets(base_cfg, tokenizer)

# ---------------------------------------------------------
# 2. 유틸리티 함수
# ---------------------------------------------------------
def compute_metrics(p):
    predictions, labels = p
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = np.argmax(predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 8e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    per_device_batch_size = 32 
    num_train_epochs = 5

    training_args = TrainingArguments(
        output_dir=f"{base_cfg['project']['output_dir']}/trial_{trial.number}",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        num_train_epochs=num_train_epochs,
        
        eval_strategy="epoch", 
        save_strategy="epoch",       
        load_best_model_at_end=True,  
        metric_for_best_model="f1",
        
        logging_steps=50,
        fp16=True, 
        report_to="none",
    )

    trainer = Trainer(
        model_init=model_init, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] 
    )

    trainer.train()
    
    metrics = trainer.evaluate()
    
    del trainer.model
    torch.cuda.empty_cache()
    
    return metrics['eval_f1']

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    
    print("Hyperparameter Tuning Started (Focusing on Weights)...")
    study.optimize(objective, n_trials=10) 

    print("------------------------------------------------")
    print("Best F1 Score:", study.best_value)
    print("Best Hyperparameters:", study.best_params)
    print("------------------------------------------------")
    