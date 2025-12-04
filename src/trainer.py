# src/trainer.py
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from src.utils import compute_metrics

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        return logits[0]
    return logits


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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg['train']['patience'])]
    )
    
    return trainer