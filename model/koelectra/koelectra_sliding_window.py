#unset HSA_OVERRIDE_GFX_VERSION
import os
import pandas as pd
import numpy as np
import torch
import warnings
import gc
import csv
import random
import shutil
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback
)
from datasets import Dataset

warnings.filterwarnings('ignore')

# ==========================================
# [1] 설정 및 하이퍼파라미터
# ==========================================

# 경로 설정
DATA_ROOT_DIR = 'make-model/data/fold'
OUTPUT_ROOT_DIR = 'make-model/model/result_models/koelectra_small_sliding_early_stopping'

# 파일명 설정
TRAIN_FILENAME = 'balanced_train.csv'
VALID_FILENAME = 'balanced_valid.csv'
TEST_FILENAME = 'local_balanced_test.csv'

# 학습 설정
N_FOLDS = 4
FOLD_DIR_PREFIX = 'fold'
MODEL_NAME = "monologg/koelectra-small-v3-discriminator"

# Sliding Window 설정
MAX_LEN = 512
STRIDE = 256

# 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 3e-5
SEED = 42
LABEL_SMOOTHING = 0.1
MAX_POOLING_THRESHOLD = 0.5 
PATIENCE = 3 

# CSV 로드 설정
detected_delimiter = ','
detected_quotechar = '"'

# ==========================================
# [2] 유틸리티 함수
# ==========================================

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

def find_column_name(columns, candidates):
    for col in columns:
        if col.lower().strip() in candidates:
            return col
    return None

def load_and_fix_data(path, is_test=False):
    if not os.path.exists(path):
        print(f" [Warning] File not found: {path}")
        return None

    df = None
    encodings_to_try = ['utf-8-sig', 'utf-8', 'cp949']

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(
                path, encoding=encoding, engine='python', on_bad_lines='skip',
                encoding_errors='ignore', delimiter=detected_delimiter,
                quotechar=detected_quotechar, quoting=csv.QUOTE_MINIMAL
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

    target_candidates = ['generated', 'label', 'target', 'class']
    target_col = find_column_name(df.columns, target_candidates)
    if target_col:
        df.rename(columns={target_col: 'label'}, inplace=True)
        try: df['label'] = df['label'].astype(int)
        except: pass
    elif not is_test:
        return None

    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)
    
    return df

# ==========================================
# [3] 문서 단위 검증 함수 (Max Pooling)
# ==========================================
def evaluate_document_level(model, df, tokenizer, device, desc="Eval"):
    """
    Validation/Test 단계에서 호출.
    데이터프레임을 순회하며 Sliding Window -> Max Pooling 수행 후 Metric 반환
    """
    model.eval()
    preds = []
    labels = []
    
    with torch.no_grad():
        iterator = tqdm(df.iterrows(), total=len(df), leave=False, desc=desc)
        
        for idx, row in iterator:
            text = str(row['text'])
            label = int(row['label']) if 'label' in row else -1
            
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                max_length=MAX_LEN, 
                stride=STRIDE, 
                truncation=True, 
                padding="max_length", 
                return_overflowing_tokens=True
            )
            
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            max_ai_prob = torch.max(probs[:, 1]).item()
            pred_label = 1 if max_ai_prob > MAX_POOLING_THRESHOLD else 0
            
            preds.append(pred_label)
            labels.append(label)
            
    if -1 not in labels:
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        return acc, f1
    else:
        return 0.0, 0.0

# ==========================================
# [4] Custom Callback Class (Early Stopping 추가)
# ==========================================
class DocumentLevelEarlyStoppingCallback(TrainerCallback):

    def __init__(self, val_df, tokenizer, device, save_dir, patience=3):
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.device = device
        self.save_dir = save_dir
        
        self.best_doc_f1 = 0.0
        self.best_model_path = os.path.join(save_dir, "best_doc_model")
        
        # Early Stopping 변수
        self.patience = patience
        self.counter = 0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # 문서 단위 평가 수행
        acc, f1 = evaluate_document_level(
            model, self.val_df, self.tokenizer, self.device, desc="Validating(Doc)"
        )
        
        print(f"\n[Epoch {state.epoch:.0f}] Document Level Valid -> Acc: {acc:.4f}, F1: {f1:.4f}")
        
        # Best Model 갱신 여부 체크
        if f1 > self.best_doc_f1:
            print(f" *** New Best Document F1! ({self.best_doc_f1:.4f} -> {f1:.4f}) Saving model...")
            self.best_doc_f1 = f1
            self.counter = 0 # 성능 향상 시 카운터 초기화
            
            if not os.path.exists(self.best_model_path):
                os.makedirs(self.best_model_path)
            
            model.save_pretrained(self.best_model_path)
            self.tokenizer.save_pretrained(self.best_model_path)
        else:
            self.counter += 1
            print(f" (No Improvement. Current Best: {self.best_doc_f1:.4f} | Patience: {self.counter}/{self.patience})")
            
            # Early Stopping 조건 달성 시
            if self.counter >= self.patience:
                print(f" [Early Stopping] Triggered! Training stopped at epoch {state.epoch:.0f}.")
                control.should_training_stop = True

# ==========================================
# [5] 메인 프로세스
# ==========================================
def run_kfold_process():
    set_seeds(SEED)
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[{MODEL_NAME}] {N_FOLDS}-Fold Training with Custom Early Stopping")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_sliding_window(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LEN,
            stride=STRIDE,
            return_overflowing_tokens=True,
            padding="max_length"
        )
        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")
        labels = []
        for i in range(len(tokenized_inputs["input_ids"])):
            labels.append(examples["label"][sample_map[i]])
        tokenized_inputs["label"] = labels
        return tokenized_inputs

    all_fold_val_metrics = []
    all_fold_test_metrics = []

    for fold_idx in range(N_FOLDS):
        print(f"\n" + "="*50)
        print(f" >>> [FOLD {fold_idx}] Start Training")
        print("="*50)

        current_fold_dir = os.path.join(DATA_ROOT_DIR, f"{FOLD_DIR_PREFIX}{fold_idx}")
        train_path = os.path.join(current_fold_dir, TRAIN_FILENAME)
        val_path = os.path.join(current_fold_dir, VALID_FILENAME)
        test_path = os.path.join(current_fold_dir, TEST_FILENAME)
        fold_output_dir = os.path.join(OUTPUT_ROOT_DIR, f"{FOLD_DIR_PREFIX}{fold_idx}")

        train_df = load_and_fix_data(train_path)
        val_df = load_and_fix_data(val_path)       
        test_df = load_and_fix_data(test_path, is_test=False)

        if train_df is None or val_df is None:
            continue

        train_ds = Dataset.from_pandas(train_df[['text', 'label']])
        val_ds = Dataset.from_pandas(val_df[['text', 'label']]) 

        print("   Processing Tokenization...")
        encoded_train = train_ds.map(preprocess_sliding_window, batched=True, remove_columns=train_ds.column_names)
        encoded_val = val_ds.map(preprocess_sliding_window, batched=True, remove_columns=val_ds.column_names)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.to(device)

        # Callback 초기화 (Patience 전달)
        doc_early_stopping_callback = DocumentLevelEarlyStoppingCallback(
            val_df=val_df,
            tokenizer=tokenizer,
            device=device,
            save_dir=fold_output_dir,
            patience=PATIENCE # Global 설정값 사용
        )

        args = TrainingArguments(
            output_dir=fold_output_dir,
            eval_strategy="no",         
            save_strategy="no",         
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            label_smoothing_factor=LABEL_SMOOTHING,
            fp16=True,
            report_to="none",
            seed=SEED,
            load_best_model_at_end=False 
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=encoded_train,
            eval_dataset=encoded_val, 
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            callbacks=[doc_early_stopping_callback] # Early Stopping Callback 등록
        )

        trainer.train()

        # ---------------------------------------------------------
        # [최종 검증] Best Model Load & Test Eval
        # ---------------------------------------------------------
        print(f"\n>>> [Fold {fold_idx}] Loading Best Document-Level Model...")
        best_model_path = os.path.join(fold_output_dir, "best_doc_model")

        if os.path.exists(best_model_path):
            best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
            best_model.to(device)
            
            val_acc, val_f1 = evaluate_document_level(best_model, val_df, tokenizer, device, desc="Final Valid")
            all_fold_val_metrics.append({"accuracy": val_acc, "f1": val_f1})
            print(f"       Valid Doc Acc: {val_acc:.4f} | Valid Doc F1: {val_f1:.4f}")

            if test_df is not None:
                test_acc, test_f1 = evaluate_document_level(best_model, test_df, tokenizer, device, desc="Final Test")
                all_fold_test_metrics.append({"accuracy": test_acc, "f1": test_f1})
                print(f"       Test Doc Acc : {test_acc:.4f} | Test Doc F1 : {test_f1:.4f}")
            else:
                print("       Test data not found.")
                all_fold_test_metrics.append({"accuracy": 0.0, "f1": 0.0})
                
            del best_model
        else:
            print(" [Error] Best model not found. Training might have failed.")
            all_fold_val_metrics.append({"accuracy": 0.0, "f1": 0.0})
            all_fold_test_metrics.append({"accuracy": 0.0, "f1": 0.0})

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    # ==========================================
    # [6] 최종 결과 요약
    # ==========================================
    print("\n" + "#"*65)
    print(" [K-Fold Final Summary: Sliding Window + Max Pooling (Early Stopping)]")
    print("#"*65)
    print(f"{'Fold':^6} | {'Valid Acc':^12} | {'Valid F1':^12} || {'Test Acc':^12} | {'Test F1':^12}")
    print("-" * 65)

    avg_val_acc = 0; avg_val_f1 = 0
    avg_test_acc = 0; avg_test_f1 = 0
    valid_count = 0; test_count = 0

    for i in range(len(all_fold_val_metrics)):
        v_acc = all_fold_val_metrics[i]['accuracy']
        v_f1 = all_fold_val_metrics[i]['f1']
        t_acc = all_fold_test_metrics[i]['accuracy']
        t_f1 = all_fold_test_metrics[i]['f1']
        
        t_str_acc = f"{t_acc:.4f}" if t_acc > 0 else "-"
        t_str_f1 = f"{t_f1:.4f}" if t_acc > 0 else "-"

        print(f"{i:^6} | {v_acc:^12.4f} | {v_f1:^12.4f} || {t_str_acc:^12} | {t_str_f1:^12}")

        if v_acc > 0:
            avg_val_acc += v_acc
            avg_val_f1 += v_f1
            valid_count += 1
        
        if t_acc > 0:
            avg_test_acc += t_acc
            avg_test_f1 += t_f1
            test_count += 1

    print("-" * 65)
    
    val_res_acc = avg_val_acc/valid_count if valid_count > 0 else 0
    val_res_f1 = avg_val_f1/valid_count if valid_count > 0 else 0
    
    print(f"{'AVG':^6} | {val_res_acc:^12.4f} | {val_res_f1:^12.4f} || ", end="")
        
    if test_count > 0:
        print(f"{avg_test_acc/test_count:^12.4f} | {avg_test_f1/test_count:^12.4f}")
    else:
        print(f"{'-':^12} | {'-':^12}")
    print("#"*65)

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_kfold_process()
    else:
        print("No GPU detected.")