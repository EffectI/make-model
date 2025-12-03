#unset HSA_OVERRIDE_GFX_VERSION
import os
import pandas as pd
import numpy as np
import torch
import warnings
import gc
import csv
import random
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset

warnings.filterwarnings('ignore')

# =========================
# [1] 설정 및 하이퍼파라미터
# =========================

# 1. 경로 설정
DATA_ROOT_DIR = 'make-model/data/fold'
OUTPUT_ROOT_DIR = 'make-model/model/result_models/koelectra_small'

# 2. 파일명 설정
TEST_FILENAME = 'local_balanced_test.csv'
TRAIN_FILENAME = 'balanced_train.csv'
VALID_FILENAME = 'balanced_valid.csv'

# 3. 폴드 설정
N_FOLDS = 4
FOLD_DIR_PREFIX = 'fold'

# 4. 모델 및 학습 설정
# 변경: KoELECTRA Small Discriminator 모델 지정
MODEL_NAME = "monologg/koelectra-small-v3-discriminator"

MAX_LEN = 512
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 3e-5
PATIENCE = 2
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
                path,
                encoding=encoding,
                engine='python',
                on_bad_lines='skip',
                encoding_errors='ignore',
                delimiter=detected_delimiter,
                quotechar=detected_quotechar,
                quoting=csv.QUOTE_MINIMAL
            )
            break
        except Exception as e:
            df = None

    if df is None:
        print(f"데이터 로드 실패: {path}")
        return None

    text_candidates = ['paragraph_text', 'text', 'sentence', 'content', 'full_text']
    text_col = find_column_name(df.columns, text_candidates)
    if text_col:
        df.rename(columns={text_col: 'text'}, inplace=True)
    else:
        obj_cols = df.select_dtypes(include=['object']).columns
        if len(obj_cols) > 0:
            df.rename(columns={obj_cols[0]: 'text'}, inplace=True)
        else:
            return None

    if is_test:
        id_candidates = ['id', 'idx', 'index', 'no', 'ID']
        id_col = find_column_name(df.columns, id_candidates)
        if id_col:
            df.rename(columns={id_col: 'id'}, inplace=True)
        else:
            df['id'] = df.index

    if not is_test:
        target_candidates = ['generated', 'label', 'target', 'class']
        target_col = find_column_name(df.columns, target_candidates)
        if target_col:
            df.rename(columns={target_col: 'label'}, inplace=True)
            try:
                df['label'] = df['label'].astype(int)
            except:
                pass
        else:
            print("Target(Label) 컬럼을 찾을 수 없습니다.")
            return None

    return df

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "f1": f1}

# ==========================================
# [3] 메인 학습 루프 (K-Fold)
# ==========================================

def run_kfold_process():
    set_seeds(SEED)
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n[{MODEL_NAME}] {N_FOLDS}-Fold 학습 시작")
    print(f"Data Root: {DATA_ROOT_DIR}")
    print(f"Output Root: {OUTPUT_ROOT_DIR}")

    # Test 데이터 로드
    test_file_path = os.path.join(DATA_ROOT_DIR, TEST_FILENAME)
    test_df = load_and_fix_data(test_file_path, is_test=False)
    
    if test_df is None:
        print("Test Set 로드 실패. 경로를 확인하세요.")
        return

    print(f"Common Test Set Loaded: {len(test_df)} samples")
    test_ds = Dataset.from_pandas(test_df[['text', 'label']])

    # AutoTokenizer가 monologg/koelectra-small-v3-discriminator를 자동으로 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN, padding=False)

    encoded_test = test_ds.map(preprocess, batched=True)

    all_fold_metrics = []

    # ==========================
    # Loop over Folds
    # ==========================
    for fold_idx in range(N_FOLDS):
        print(f"\n" + "="*40)
        print(f" >>> [FOLD {fold_idx}] Start Training")
        print("="*40)

        current_fold_dir = os.path.join(DATA_ROOT_DIR, f"{FOLD_DIR_PREFIX}{fold_idx}")
        train_path = os.path.join(current_fold_dir, TRAIN_FILENAME)
        val_path = os.path.join(current_fold_dir, VALID_FILENAME)
        fold_output_dir = os.path.join(OUTPUT_ROOT_DIR, f"{FOLD_DIR_PREFIX}{fold_idx}")

        print(f" - Train: {train_path}")
        print(f" - Valid: {val_path}")
        print(f" - Output: {fold_output_dir}")

        train_df = load_and_fix_data(train_path)
        val_df = load_and_fix_data(val_path)

        if train_df is None or val_df is None:
            print(f"!! [Fold {fold_idx}] 데이터 로드 실패. 건너뜁니다.")
            continue

        train_ds = Dataset.from_pandas(train_df[['text', 'label']])
        val_ds = Dataset.from_pandas(val_df[['text', 'label']])

        encoded_train = train_ds.map(preprocess, batched=True)
        encoded_val = val_ds.map(preprocess, batched=True)

        # AutoModel이 Electra 아키텍처를 자동으로 로드합니다.
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

        args = TrainingArguments(
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
        
        # 모델 저장
        trainer.save_model(fold_output_dir)
        tokenizer.save_pretrained(fold_output_dir)

        # -----------------------
        # 기본 평가 (Metric 계산)
        # -----------------------
        print(f">>> [Fold {fold_idx}] Evaluating on Test Set...")
        metrics = trainer.evaluate(encoded_test)
        print(f"    Result: {metrics}")
        all_fold_metrics.append(metrics)

        # ---------------------------------------
        # 상세 예측 결과 저장 (Correlation 분석용)
        # ---------------------------------------
        print(f">>> [Fold {fold_idx}] Saving Predictions for Correlation Analysis...")
        
        pred_output = trainer.predict(encoded_test)
        logits = pred_output.predictions
        
        # Softmax를 적용하여 확률값(Probability) 추출
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        
        fold_pred_df = test_df.copy()
        
        # 텍스트 컬럼 제거 원할 시 주석 해제
        # fold_pred_df = fold_pred_df.drop(columns=['text']) 
        
        fold_pred_df['prob_0'] = probs[:, 0]  # Class 0일 확률
        fold_pred_df['prob_1'] = probs[:, 1]  # Class 1일 확률
        fold_pred_df['pred_label'] = np.argmax(probs, axis=1) # 최종 예측 라벨
        
        # CSV 저장
        pred_save_path = os.path.join(fold_output_dir, f"{FOLD_DIR_PREFIX}{fold_idx}_predictions.csv")
        fold_pred_df.to_csv(pred_save_path, index=False)
        print(f"    Saved: {pred_save_path}")

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

    # ==========================
    # Final Summary
    # ==========================
    print("\n" + "#"*50)
    print(" [K-Fold Training Summary]")
    print("#"*50)

    avg_acc = 0
    avg_f1 = 0

    for i, m in enumerate(all_fold_metrics):
        print(f" Fold {i} -> Accuracy: {m['eval_accuracy']:.4f}, F1: {m['eval_f1']:.4f}")
        avg_acc += m['eval_accuracy']
        avg_f1 += m['eval_f1']

    if len(all_fold_metrics) > 0:
        print("-" * 50)
        print(f" Average -> Accuracy: {avg_acc/len(all_fold_metrics):.4f}, F1: {avg_f1/len(all_fold_metrics):.4f}")
    print("#"*50)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        run_kfold_process()
    else:
        print("No GPU detected.")