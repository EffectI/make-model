#unset HSA_OVERRIDE_GFX_VERSION
import os
import pandas as pd
import numpy as np
import torch
import warnings
import gc
import csv
import random
from sklearn.metrics import f1_score, accuracy_score
<<<<<<< HEAD
=======
from sklearn.model_selection import train_test_split # 데이터 분리를 위해 추가
>>>>>>> koelectra-truing
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset

<<<<<<< HEAD

=======
>>>>>>> koelectra-truing
warnings.filterwarnings('ignore')

# =========================
# [1] 설정 및 하이퍼파라미터
# =========================

<<<<<<< HEAD
DATA_ROOT_DIR = 'make-model/data/fold'
OUTPUT_ROOT_DIR = 'make-model/model/result_models/koelectra_small_sliding'

TEST_FILENAME = 'local_balanced_test.csv'
TRAIN_FILENAME = 'balanced_train.csv'
VALID_FILENAME = 'balanced_valid.csv'

N_FOLDS = 4
FOLD_DIR_PREFIX = 'fold'
=======
# 기존 Fold 관련 경로 주석 처리
# DATA_ROOT_DIR = 'make-model/data/fold'
# OUTPUT_ROOT_DIR = 'make-model/model/result_models/koelectra_small_sliding'

# 단일 학습용 경로 설정
DATA_FILE_PATH = 'make-model/data/interim/clean_train_balanced.csv'
OUTPUT_DIR = 'make-model/model/result_models/koelectra_small_sliding_single'

# Fold 설정 주석 처리
# N_FOLDS = 4
# FOLD_DIR_PREFIX = 'fold'
# TEST_FILENAME = 'local_balanced_test.csv'
# TRAIN_FILENAME = 'balanced_train.csv'
# VALID_FILENAME = 'balanced_valid.csv'

>>>>>>> koelectra-truing
MODEL_NAME = "monologg/koelectra-small-v3-discriminator"

# 슬라이딩 윈도우 설정
MAX_LEN = 512
STRIDE = 256 

BATCH_SIZE = 32 
EPOCHS = 10
LEARNING_RATE = 3e-5
PATIENCE = 3
SEED = 42
<<<<<<< HEAD
=======
VAL_SIZE = 0.2 # 전체 데이터 중 20%를 검증용으로 사용
>>>>>>> koelectra-truing

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
<<<<<<< HEAD
    # 전체 문서 단위 정확도와는 다를 수 있습니다.
=======
>>>>>>> koelectra-truing
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "f1": f1}

# ==========================================
<<<<<<< HEAD
# [3] 메인 학습 루프
# ==========================================

def run_kfold_process():
=======
# [3] 메인 학습 루프 (K-Fold 제거됨)
# ==========================================

# 함수명을 run_kfold_process -> run_training으로 변경
def run_training():
>>>>>>> koelectra-truing
    set_seeds(SEED)
    torch.cuda.empty_cache()
    gc.collect()

<<<<<<< HEAD
    print(f"\n[{MODEL_NAME}] {N_FOLDS}-Fold Sliding Window 학습 시작")
    print(f"Data Root: {DATA_ROOT_DIR}")
    print(f"Output Root: {OUTPUT_ROOT_DIR}")
=======
    print(f"\n[{MODEL_NAME}] 단일 학습(Single Run) 시작")
    print(f"Data File: {DATA_FILE_PATH}")
    print(f"Output Dir: {OUTPUT_DIR}")
>>>>>>> koelectra-truing

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ----------------------------------------------------------------
    # [핵심 함수] 슬라이딩 윈도우 전처리
<<<<<<< HEAD
    # 하나의 긴 텍스트를 여러 개의 샘플로 쪼개고, 라벨을 복사합니다.
=======
>>>>>>> koelectra-truing
    # ----------------------------------------------------------------
    def preprocess_sliding_window(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LEN,
            stride=STRIDE,                  # 겹치는 구간 설정
            return_overflowing_tokens=True, # 긴 문서를 여러 개로 쪼개서 반환
            padding="max_length"            # 배치 처리를 위해 패딩
        )

<<<<<<< HEAD
        # 쪼개진 조각들이 원래 어떤 샘플(문서)에서 왔는지 매핑 정보
        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")
        
        # 라벨 복사 작업
        labels = []
        for i in range(len(tokenized_inputs["input_ids"])):
            # 현재 조각(chunk)이 유래한 원본 문서의 인덱스
            sample_idx = sample_map[i]
            # 원본 문서의 라벨을 가져와서 현재 조각에 부여
=======
        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping")
        
        labels = []
        for i in range(len(tokenized_inputs["input_ids"])):
            sample_idx = sample_map[i]
>>>>>>> koelectra-truing
            labels.append(examples["label"][sample_idx])
            
        tokenized_inputs["label"] = labels
        return tokenized_inputs

<<<<<<< HEAD
    # Test 데이터 로드 (평가용) - Test도 윈도우 방식으로 자릅니다.
    # (실제 추론 시에는 Max Pooling을 해야 하지만, 학습 중 Metric 확인용으로는 Chunk 단위 평가를 수행합니다)
    test_file_path = os.path.join(DATA_ROOT_DIR, TEST_FILENAME)
    test_df = load_and_fix_data(test_file_path, is_test=False) # 학습 중 평가를 위해 라벨이 있는 Test셋 사용 가정
    
    if test_df is not None:
        test_ds = Dataset.from_pandas(test_df[['text', 'label']])
        # batched=True와 remove_columns가 필수입니다. (입력 행 수 != 출력 행 수 이기 때문)
        encoded_test = test_ds.map(
            preprocess_sliding_window, 
            batched=True, 
            remove_columns=test_ds.column_names
        )
        print(f"Test Set Expanded: {len(test_df)} docs -> {len(encoded_test)} chunks")
    else:
        encoded_test = None

    all_fold_metrics = []

    for fold_idx in range(N_FOLDS):
        print(f"\n" + "="*40)
        print(f" >>> [FOLD {fold_idx}] Start Training")
        print("="*40)

        current_fold_dir = os.path.join(DATA_ROOT_DIR, f"{FOLD_DIR_PREFIX}{fold_idx}")
        train_path = os.path.join(current_fold_dir, TRAIN_FILENAME)
        val_path = os.path.join(current_fold_dir, VALID_FILENAME)
        fold_output_dir = os.path.join(OUTPUT_ROOT_DIR, f"{FOLD_DIR_PREFIX}{fold_idx}")

        train_df = load_and_fix_data(train_path)
        val_df = load_and_fix_data(val_path)

        if train_df is None or val_df is None:
            continue

        train_ds = Dataset.from_pandas(train_df[['text', 'label']])
        val_ds = Dataset.from_pandas(val_df[['text', 'label']])

        # ========================================================
        # remove_columns를 반드시 해서 원본 텍스트 컬럼을 날려야 함
        # ========================================================
        print("   Applying Sliding Window Tokenization...")
        encoded_train = train_ds.map(
            preprocess_sliding_window, 
            batched=True, 
            remove_columns=train_ds.column_names
        )
        encoded_val = val_ds.map(
            preprocess_sliding_window, 
            batched=True, 
            remove_columns=val_ds.column_names
        )
        
        print(f"   Train: {len(train_df)} -> {len(encoded_train)} chunks")
        print(f"   Valid: {len(val_df)} -> {len(encoded_val)} chunks")

        # 모델 초기화
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        


        args = TrainingArguments(
            label_smoothing_factor=0.1,
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
            train_dataset=encoded_train, eval_dataset=encoded_val,
            tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
        )

        trainer.train()
        
        trainer.save_model(fold_output_dir)
        tokenizer.save_pretrained(fold_output_dir)

        # 평가 (Chunk 단위 평가임에 유의)
        if encoded_test:
            metrics = trainer.evaluate(encoded_test)
            print(f"    Chunk-level Result: {metrics}")
            all_fold_metrics.append(metrics)

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "#"*50)
    print(" [K-Fold Training Summary]")
    print("#"*50)
    # 생략 (Chunk 레벨 결과 평균 출력)  

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_kfold_process()
=======
    # 1. 단일 데이터 파일 로드
    full_df = load_and_fix_data(DATA_FILE_PATH)
    if full_df is None:
        print("데이터 로드 실패. 종료합니다.")
        return

    # 2. Train / Validation 분리 (단일 파일이므로 직접 분리 필요)
    print(f"전체 데이터 개수: {len(full_df)}")
    train_df, val_df = train_test_split(
        full_df, 
        test_size=VAL_SIZE, 
        random_state=SEED, 
        stratify=full_df['label'] # 라벨 비율 유지
    )

    print(f"학습 데이터(Train): {len(train_df)}")
    print(f"검증 데이터(Valid): {len(val_df)}")

    # 3. Dataset 변환
    train_ds = Dataset.from_pandas(train_df[['text', 'label']])
    val_ds = Dataset.from_pandas(val_df[['text', 'label']])

    # 4. 슬라이딩 윈도우 적용
    print(" Applying Sliding Window Tokenization...")
    encoded_train = train_ds.map(
        preprocess_sliding_window, 
        batched=True, 
        remove_columns=train_ds.column_names
    )
    encoded_val = val_ds.map(
        preprocess_sliding_window, 
        batched=True, 
        remove_columns=val_ds.column_names
    )
    
    print(f" Expanded Train Chunks: {len(encoded_train)}")
    print(f" Expanded Valid Chunks: {len(encoded_val)}")

    # 5. 모델 초기화
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # K-Fold Loop 제거됨 (주석 처리 예시)
    # for fold_idx in range(N_FOLDS):
    #     ... (기존 코드 생략) ...

    # 6. Trainer 설정
    args = TrainingArguments(
        label_smoothing_factor=0.1,
        output_dir=OUTPUT_DIR,
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
        train_dataset=encoded_train, eval_dataset=encoded_val,
        tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )

    # 7. 학습 시작
    trainer.train()
    
    # 8. 최종 모델 저장
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 평가 결과 출력
    final_metrics = trainer.evaluate()
    print(f"Final Validation Metrics: {final_metrics}")

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "#"*50)
    print(" [Training Finished]")
    print("#"*50)

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_training()
>>>>>>> koelectra-truing
    else:
        print("No GPU detected.")