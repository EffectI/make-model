#unset HSA_OVERRIDE_GFX_VERSION
import os
import pandas as pd
import numpy as np
import torch
import warnings
import gc
import csv
import random
from tqdm import tqdm  # 진행상황 표시를 위해 추가
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset

# LoRA 설정이 필요하다면 주석 해제 후 사용 (현재는 Base Finetuning 기준)
# from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings('ignore')

# =========================
# [1] 설정 및 하이퍼파라미터
# =========================

DATA_ROOT_DIR = 'make-model/data/fold'
OUTPUT_ROOT_DIR = 'make-model/model/result_models/koelectra_small_sliding'

TEST_FILENAME = 'local_balanced_test.csv'
TRAIN_FILENAME = 'balanced_train.csv'
VALID_FILENAME = 'balanced_valid.csv'

N_FOLDS = 4
FOLD_DIR_PREFIX = 'fold'
MODEL_NAME = "monologg/koelectra-small-v3-discriminator"

MAX_LEN = 512
STRIDE = 256  # 윈도우가 겹치는 길이 (보통 MAX_LEN의 50% 설정)

BATCH_SIZE = 32 
EPOCHS = 10
LEARNING_RATE = 3e-5
PATIENCE = 3
SEED = 42

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
        print(f"파일이 없습니다: {path}")
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

    if not is_test:
        target_candidates = ['generated', 'label', 'target', 'class']
        target_col = find_column_name(df.columns, target_candidates)
        if target_col:
            df.rename(columns={target_col: 'label'}, inplace=True)
            try: df['label'] = df['label'].astype(int)
            except: pass
        else: return None

    # 결측치 제거
    df = df.dropna(subset=['text'])
    # 텍스트 문자열 변환
    df['text'] = df['text'].astype(str)
    
    return df

def compute_metrics(p):
    # 주의: 여기서 계산되는 Accuracy는 'Chunk(조각)' 단위의 정확도입니다.
    # 전체 문서 단위 정확도와는 다를 수 있습니다.
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "f1": f1}

# ==========================================
# [3] 문서 단위 검증 함수 (새로 추가됨)
# ==========================================
def validate_document_level(model, val_df, tokenizer, device):
    """
    Validation Set 원본(문서 단위)을 받아서, 
    슬라이딩 윈도우 + Max Pooling으로 진짜 성능을 측정하는 함수
    """
    model.eval()
    preds = []
    labels = []
    
    print("    -> Validating on Document Level (Max Pooling)...")
    
    with torch.no_grad():
        # tqdm으로 진행상황 표시
        for idx, row in tqdm(val_df.iterrows(), total=len(val_df), leave=False):
            text = str(row['text'])
            label = int(row['label'])
            
            # 슬라이딩 윈도우 토큰화 (즉석에서 수행)
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                max_length=MAX_LEN, 
                stride=STRIDE, 
                truncation=True, 
                padding="max_length", 
                return_overflowing_tokens=True
            )
            
            # GPU로 이동. shape: [num_chunks, 512]
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # 모델 예측
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 확률 변환
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # [핵심 로직] Max Pooling
            # 여러 청크 중 하나라도 AI(1) 확률이 0.5를 넘으면, 문서를 AI로 판단
            # 또는 AI 클래스 확률의 최댓값을 기준으로 판단
            max_ai_prob = torch.max(probs[:, 1]).item()
            
            # 최종 예측 라벨 결정
            pred_label = 1 if max_ai_prob > 0.5 else 0
            
            preds.append(pred_label)
            labels.append(label)
            
    # Metric 계산
    doc_acc = accuracy_score(labels, preds)
    doc_f1 = f1_score(labels, preds, average='macro')
    
    return doc_acc, doc_f1

# ==========================================
# [4] 메인 학습 루프
# ==========================================

def run_kfold_process():
    set_seeds(SEED)
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[{MODEL_NAME}] {N_FOLDS}-Fold Sliding Window 학습 시작")
    print(f"Data Root: {DATA_ROOT_DIR}")
    print(f"Output Root: {OUTPUT_ROOT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ----------------------------------------------------------------
    # [핵심 함수] 슬라이딩 윈도우 전처리 (학습용)
    # ----------------------------------------------------------------
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
            sample_idx = sample_map[i]
            labels.append(examples["label"][sample_idx])
            
        tokenized_inputs["label"] = labels
        return tokenized_inputs

    # Test 데이터 로드 (필요시 사용, 여기서는 생략 가능)
    # test_file_path = os.path.join(DATA_ROOT_DIR, TEST_FILENAME)
    # test_df = load_and_fix_data(test_file_path, is_test=False) 
    
    # if test_df is not None:
    #     test_ds = Dataset.from_pandas(test_df[['text', 'label']])
    #     encoded_test = test_ds.map(
    #         preprocess_sliding_window, 
    #         batched=True, 
    #         remove_columns=test_ds.column_names
    #     )
    # else:
    #     encoded_test = None
    encoded_test = None # 간단하게 처리

    all_fold_doc_metrics = [] # 문서 단위 결과를 저장할 리스트

    for fold_idx in range(N_FOLDS):
        print(f"\n" + "="*40)
        print(f" >>> [FOLD {fold_idx}] Start Training")
        print("="*40)

        current_fold_dir = os.path.join(DATA_ROOT_DIR, f"{FOLD_DIR_PREFIX}{fold_idx}")
        train_path = os.path.join(current_fold_dir, TRAIN_FILENAME)
        val_path = os.path.join(current_fold_dir, VALID_FILENAME)
        fold_output_dir = os.path.join(OUTPUT_ROOT_DIR, f"{FOLD_DIR_PREFIX}{fold_idx}")

        # 원본 데이터 로드 (검증 시 사용하기 위해 변수 유지)
        train_df = load_and_fix_data(train_path)
        val_df = load_and_fix_data(val_path)

        if train_df is None or val_df is None:
            continue

        train_ds = Dataset.from_pandas(train_df[['text', 'label']])
        val_ds = Dataset.from_pandas(val_df[['text', 'label']])

        # 학습용 데이터셋 매핑 (Sliding Window 적용)
        print("   Applying Sliding Window Tokenization for Training...")
        encoded_train = train_ds.map(
            preprocess_sliding_window, 
            batched=True, 
            remove_columns=train_ds.column_names
        )
        encoded_val_chunk = val_ds.map(
            preprocess_sliding_window, 
            batched=True, 
            remove_columns=val_ds.column_names
        )
        
        print(f"   Train Chunks: {len(encoded_train)} | Valid Chunks: {len(encoded_val_chunk)}")

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.to(device)

        args = TrainingArguments(
            label_smoothing_factor=0.1, # 라벨 노이즈 완화
            output_dir=fold_output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            seed=SEED
        )

        trainer = Trainer(
            model=model, args=args,
            train_dataset=encoded_train, 
            eval_dataset=encoded_val_chunk, # Trainer는 청크 단위로 검증하며 Loss를 계산함
            tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
        )

        trainer.train()
        
        trainer.save_model(fold_output_dir)
        tokenizer.save_pretrained(fold_output_dir)

        # ----------------------------------------------------------------
        # [핵심] 학습 완료 후, 진짜 문서 단위(Document Level) 성능 확인
        # ----------------------------------------------------------------
        print(f"\n>>> [Fold {fold_idx}] Calculating REAL Document-Level Metrics...")
        
        # 원본 val_df를 사용하여 검증
        doc_acc, doc_f1 = validate_document_level(model, val_df, tokenizer, device)
        
        print(f"    📊 Real Document Accuracy: {doc_acc:.4f}")
        print(f"    📊 Real Document F1-Score: {doc_f1:.4f}")
        print("-" * 40)
        
        all_fold_doc_metrics.append({"accuracy": doc_acc, "f1": doc_f1})

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "#"*50)
    print(" [K-Fold Real Document-Level Summary]")
    print("#"*50)
    
    avg_acc = sum([m['accuracy'] for m in all_fold_doc_metrics]) / len(all_fold_doc_metrics)
    avg_f1 = sum([m['f1'] for m in all_fold_doc_metrics]) / len(all_fold_doc_metrics)
    
    for i, m in enumerate(all_fold_doc_metrics):
        print(f" Fold {i} -> Doc Acc: {m['accuracy']:.4f}, Doc F1: {m['f1']:.4f}")
    
    print("-" * 50)
    print(f" Average -> Doc Acc: {avg_acc:.4f}, Doc F1: {avg_f1:.4f}")
    print("#"*50)

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_kfold_process()
    else:
        print("No GPU detected.")