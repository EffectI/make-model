# src/data_loader.py
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

def find_column_name(columns, candidates):
    for col in columns:
        if col.lower().strip() in candidates:
            return col
    return None

def load_and_fix_data(path, delimiter=',', quotechar='"', is_test=False):
    if not os.path.exists(path):
        print(f"파일이 없습니다: {path}")
        return None

    df = None
    encodings_to_try = ['utf-8-sig', 'utf-8', 'cp949']

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(
                path, encoding=encoding, engine='python', on_bad_lines='skip',
                encoding_errors='ignore', delimiter=delimiter,
                quotechar=quotechar, quoting=csv.QUOTE_MINIMAL
            )
            break
        except:
            df = None

    if df is None: return None

    text_candidates = ['paragraph_text', 'text', 'sentence', 'content', 'full_text']
    text_col = find_column_name(df.columns, text_candidates)
    if text_col:
        df.rename(columns={text_col: 'text'}, inplace=True)
    else:
        obj_cols = df.select_dtypes(include=['object']).columns
        if len(obj_cols) > 0: df.rename(columns={obj_cols[0]: 'text'}, inplace=True)
        else: return None

    if is_test:
        id_candidates = ['id', 'idx', 'index', 'no', 'ID']
        id_col = find_column_name(df.columns, id_candidates)
        if id_col: df.rename(columns={id_col: 'id'}, inplace=True)
        else: df['id'] = df.index

    if not is_test:
        target_candidates = ['generated', 'label', 'target', 'class']
        target_col = find_column_name(df.columns, target_candidates)
        if target_col:
            df.rename(columns={target_col: 'label'}, inplace=True)
            try: df['label'] = df['label'].astype(int)
            except: pass
        else: return None

    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)
    return df


def get_tokenized_datasets(cfg, tokenizer):
    full_df = load_and_fix_data(
        cfg['data']['file_path'], 
        delimiter=cfg['data']['delimiter'],
        quotechar=cfg['data']['quotechar']
    )
    if full_df is None: raise ValueError("Data load failed")

    train_df, val_df = train_test_split(
        full_df, 
        test_size=cfg['data']['val_size'], 
        random_state=cfg['project']['seed'], 
        stratify=full_df['label']
    )
    
    train_ds = Dataset.from_pandas(train_df[['text', 'label']])
    val_ds = Dataset.from_pandas(val_df[['text', 'label']])


    def preprocess_sliding(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg['data']['max_len'],
            stride=cfg['data']['stride'],
            return_overflowing_tokens=True, 
            padding="max_length"
        )
        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")
        labels = []
        for i in range(len(tokenized_inputs["input_ids"])):
            sample_idx = sample_map[i]
            labels.append(examples["label"][sample_idx])
        tokenized_inputs["label"] = labels
        return tokenized_inputs

    def preprocess_basic(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True, 
            max_length=cfg['data']['max_len'],
            padding="max_length"
        )
        tokenized_inputs["label"] = examples["label"]
        return tokenized_inputs

    method = cfg['data'].get('method', 'basic') 
    
    print(f"Applying Tokenization Strategy: [{method.upper()}]")
    
    if method == 'sliding':
        preprocess_fn = preprocess_sliding
        remove_cols = train_ds.column_names 
    else:
        preprocess_fn = preprocess_basic
        remove_cols = ['text'] 

    encoded_train = train_ds.map(preprocess_fn, batched=True, remove_columns=remove_cols)
    encoded_val = val_ds.map(preprocess_fn, batched=True, remove_columns=remove_cols)

    print(f"Train Size: {len(encoded_train)}")
    print(f"Valid Size: {len(encoded_val)}")

    return encoded_train, encoded_val