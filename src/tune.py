import os
import yaml
import torch
import numpy as np
import optuna
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_loader import get_tokenized_datasets

CONFIG_PATH = "config/roberta.yaml" 
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    base_cfg = yaml.safe_load(f)


MODEL_NAME = base_cfg['model']['name']
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("데이터셋 로드 중...")
train_dataset, eval_dataset = get_tokenized_datasets(base_cfg, tokenizer)

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
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 4e-5, log=True)
    #num_train_epochs = trial.suggest_int("num_train_epochs", 3, 5)
    #weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    #batch_size = trial.suggest_categorical("batch_size", [16, 32])


    training_args = TrainingArguments(
        output_dir=f"{base_cfg['project']['output_dir']}/trial_{trial.number}",
        learning_rate=learning_rate,
        #per_device_train_batch_size=batch_size,
        #per_device_eval_batch_size=batch_size,
        #num_train_epochs=num_train_epochs,
        #weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="no",
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
    )

    trainer.train()
    
    metrics = trainer.evaluate()
    
    return metrics['eval_f1']

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    
    print("Hyperparameter Tuning Started...")
    study.optimize(objective, n_trials=10) 

    print("------------------------------------------------")
    print("Best F1 Score:", study.best_value)
    print("Best Hyperparameters:", study.best_params)
    print("------------------------------------------------")