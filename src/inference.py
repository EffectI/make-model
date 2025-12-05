# src/inference.py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import DataCollatorWithPadding
from datasets import Dataset

class Predictor:
    def __init__(self, model, tokenizer, device=None):

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model.eval() 

    def predict(self, texts, batch_size=32, max_len=512):
        dataset = Dataset.from_dict({'text': texts})
        def preprocess(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_len,
                padding="max_length"
            )

        encoded_ds = dataset.map(preprocess, batched=True, remove_columns=['text'])
        
        data_collator = DataCollatorWithPadding(self.tokenizer)
        dataloader = DataLoader(
            encoded_ds, 
            batch_size=batch_size, 
            collate_fn=data_collator,
            shuffle=False 
        )

        all_probs = []
        print(f"Start Inference on {len(texts)} samples...")
        
        with torch.no_grad():
                    for batch in tqdm(dataloader):
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        
                        if hasattr(outputs, "logits"):
                            logits = outputs.logits
                        elif isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs
                        
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        all_probs.append(probs.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        preds = np.argmax(all_probs, axis=1)
        
        return preds, all_probs