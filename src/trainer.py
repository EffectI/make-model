import torch
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from src.utils import compute_metrics


def preprocess_logits_for_metrics(logits, labels):

    if hasattr(logits, "logits"):
        logits = logits.logits
    
    elif isinstance(logits, tuple):
        if len(logits) == 0:
            return torch.zeros(labels.shape[0], device=labels.device, dtype=torch.long)
            
        found_tensor = None
        for item in logits:
            if isinstance(item, torch.Tensor) and item.dim() == 2:
                found_tensor = item
                break
        
        if found_tensor is not None:
            logits = found_tensor
        elif len(logits) > 1:
            logits = logits[1]
        else:
            logits = logits[0]

    elif isinstance(logits, dict):
        if 'logits' in logits:
            logits = logits['logits']
    
    if isinstance(logits, torch.Tensor):
        return logits.argmax(dim=-1)
        
    return torch.zeros(labels.shape[0], device=labels.device, dtype=torch.long)


def get_trainer(model, tokenizer, train_ds, val_ds, cfg):
    args = TrainingArguments(
        output_dir=cfg['project']['output_dir'],
        label_smoothing_factor=cfg['train']['label_smoothing'],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(cfg['train']['learning_rate']),
        per_device_train_batch_size=cfg['train']['batch_size'],
        per_device_eval_batch_size=cfg['train']['batch_size'],
        num_train_epochs=cfg['train']['epochs'],
        weight_decay=cfg['train']['weight_decay'],
        fp16=cfg['train']['fp16'],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=cfg['project']['seed']
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics, # 함수 연결
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg['train']['patience'])]
    )
    
    return trainer