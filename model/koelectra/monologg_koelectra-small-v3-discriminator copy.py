import os
import pandas as pd
import numpy as np
import torch
import warnings
import gc
import csv
import random
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

warnings.filterwarnings('ignore')

# ==========================================
# [1] 설정 및 하이퍼파라미터
# ==========================================

# 1. 경로 설정 (요청한 경로 적용함)
# 현재 실행 위치 기준 상대 경로로 설정함
DATA_ROOT_DIR = 'make-model/data/fold'
OUTPUT_ROOT_DIR = 'make-model/model/result_models/gemma_2b_it_4fold'

# 2. 파일명 설정
TEST_FILENAME = 'local_origin_test.csv'
TRAIN_FILENAME = 'origin_train.csv'
VALID_FILENAME = 'origin_valid.csv'

# 3. 폴드 설정
N_FOLDS = 4
FOLD_DIR_PREFIX = 'fold'

# 4. 모델 및 학습 설정 (Gemma용)
# 사용하려는 실제 모델 ID로 변경 필요함
MODEL_NAME = "google/gemma-2-2b-it"

# LLM은 메모리를 많이 차지하므로 배치 사이즈를 작게 설정해야 함
MAX_LEN = 512
BATCH_SIZE = 2       # 32GB RAM/VRAM 고려하여 2~4로 설정함 (OOM 발생 시 1로 줄일 것)
GRAD_ACCUMULATION = 4 # 배치 사이즈가 작으므로 그라디언트 누적을 사용함
EPOCHS = 3           # LLM은 보통 적은 에폭으로도 학습됨
LEARNING_RATE = 2e-5 # LLM Fine-tuning 권장 학습률
PATIENCE = 2         # Early Stopping
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
        print(f"파일이 없음: {path}")
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
        except Exception:
            df = None

    if df is None:
        print(f"데이터 로드 실패함: {path}")
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
            print("Target(Label) 컬럼을 찾을 수 없음")
            return None

    return df

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    # Binary classification이므로 average='binary' 혹은 'macro' 선택
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "f1": f1}

# ==========================================
# [3] 메인 학습 루프 (K-Fold)
# ==========================================

def run_kfold_process():
    set_seeds(SEED)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\n[{MODEL_NAME}] {N_FOLDS}-Fold 학습 시작함")
    print(f"Data Root: {DATA_ROOT_DIR}")
    print(f"Output Root: {OUTPUT_ROOT_DIR}")

    # 1. Test 데이터 로드 (공통)
    test_file_path = os.path.join(DATA_ROOT_DIR, TEST_FILENAME)
    # is_test=False로 로드하여 label이 있다면 포함하도록 함 (검증용)
    test_df = load_and_fix_data(test_file_path, is_test=False)
    
    if test_df is None:
        print("Test Set 로드 실패함. 경로 확인 필요함")
        return

    print(f"Common Test Set Loaded: {len(test_df)} samples")
    test_ds = Dataset.from_pandas(test_df)

    # 2. 토크나이저 설정 (Gemma 특화)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"토크나이저 로드 실패함: {e}")
        return

    # Gemma/Llama 계열은 pad_token이 없는 경우가 많아 eos_token으로 대체함
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad Token이 없어 EOS Token으로 설정함")

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

        # 폴드별 출력 디렉토리 생성
        os.makedirs(fold_output_dir, exist_ok=True)

        print(f" - Train: {train_path}")
        print(f" - Valid: {val_path}")
        print(f" - Output: {fold_output_dir}")

        train_df = load_and_fix_data(train_path)
        val_df = load_and_fix_data(val_path)

        if train_df is None or val_df is None:
            print(f"!! [Fold {fold_idx}] 데이터 로드 실패함. 건너뜀")
            continue

        train_ds = Dataset.from_pandas(train_df[['text', 'label']])
        val_ds = Dataset.from_pandas(val_df[['text', 'label']])

        encoded_train = train_ds.map(preprocess, batched=True)
        encoded_val = val_ds.map(preprocess, batched=True)

        # 3. 모델 로드 (Gemma Classification Head)
        # device_map="auto"를 사용하여 GPU 메모리에 맞게 자동 할당함
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=2,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        # 모델 패딩 토큰 ID 동기화
        model.config.pad_token_id = tokenizer.pad_token_id

        # 4. 학습 인자 설정
        args = TrainingArguments(
            output_dir=fold_output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION, # 작은 배치 사이즈 보완
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(), # GPU 사용 시 fp16
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            seed=SEED,
            gradient_checkpointing=True # 메모리 절약 필수
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
        # 평가 (Metric 계산)
        # -----------------------
        print(f">>> [Fold {fold_idx}] Evaluating on Test Set...")
        metrics = trainer.evaluate(encoded_test)
        print(f"    Result: {metrics}")
        all_fold_metrics.append(metrics)

        # -----------------------
        # 상세 예측 결과 저장
        # -----------------------
        print(f">>> [Fold {fold_idx}] Saving Predictions...")
        
        pred_output = trainer.predict(encoded_test)
        logits = pred_output.predictions
        
        # Softmax 확률 변환
        # logits가 numpy 배열일 수도 있고 torch tensor일 수도 있음
        if isinstance(logits, np.ndarray):
            logits = torch.tensor(logits)
        
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
        
        fold_pred_df = test_df.copy()
        
        fold_pred_df['prob_0'] = probs[:, 0]
        fold_pred_df['prob_1'] = probs[:, 1]
        fold_pred_df['pred_label'] = np.argmax(probs, axis=1)
        
        pred_save_path = os.path.join(fold_output_dir, f"{FOLD_DIR_PREFIX}{fold_idx}_predictions.csv")
        fold_pred_df.to_csv(pred_save_path, index=False)
        print(f"    Saved: {pred_save_path}")

        # 메모리 정리
        del model, trainer
        if torch.cuda.is_available():
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
        acc = m.get('eval_accuracy', 0)
        f1 = m.get('eval_f1', 0)
        print(f" Fold {i} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
        avg_acc += acc
        avg_f1 += f1

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