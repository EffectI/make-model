# src/model.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_model(model_name, num_labels=2):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)