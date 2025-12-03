import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# =========================
# [1] 설정
# =========================

# 학습된 모델이 저장된 경로
MODEL_DIR = 'make-model/experiment/koelectra_small_sliding_single'

# 테스트 데이터 경로
TEST_DATA_PATH = 'make-model/data/raw/test.csv'

# 결과 저장 경로
SUBMISSION_PATH = 'make-model/experiments/temp_submission/submission.csv'

# 하이퍼파라미터
MAX_LEN = 512
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# [2] 유틸리티 클래스/함수
# =========================

class TestDataset(Dataset):
    def __init__(self, tokenized_inputs):
        self.input_ids = tokenized_inputs['input_ids']
        self.attention_mask = tokenized_inputs['attention_mask']

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.input_ids)

def load_test_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    
    df = None
    for encoding in ['utf-8-sig', 'utf-8', 'cp949']:
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except:
            continue
    
    if df is None:
        raise ValueError("CSV 파일을 읽을 수 없습니다.")

    # 텍스트 컬럼 찾기
    text_col = None
    for col in df.columns:
        if col.lower() in ['text', 'content', 'paragraph_text']:
            text_col = col
            break
            
    if text_col is None:
        obj_cols = df.select_dtypes(include=['object']).columns
        if len(obj_cols) > 0:
            text_col = obj_cols[0]
        else:
            raise ValueError("텍스트 컬럼을 찾을 수 없습니다.")

    print(f"텍스트 컬럼 감지됨: {text_col}")
    
    # id 컬럼 찾기
    id_col = 'id'
    if 'id' not in df.columns:
        candidates = ['ID', 'idx', 'index', 'no']
        for cand in candidates:
            if cand in df.columns:
                id_col = cand
                break
    
    return df, text_col, id_col

# =========================
# [3] 메인 추론 로직
# =========================

def main():
    print(f"Device: {DEVICE}")
    print("Loading Model & Tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()

    print("Loading Test Data...")
    test_df, text_col, id_col = load_test_data(TEST_DATA_PATH)
    
    test_df[text_col] = test_df[text_col].fillna("").astype(str)
    texts = test_df[text_col].tolist()

    # 1. 단순 토크나이징 (슬라이딩 윈도우 제거, 512길이 초과시 잘라냄)
    print("Tokenizing (Truncation)...")
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors=None # 리스트 형태로 반환받아 Dataset에서 텐서 변환
    )
    
    test_dataset = TestDataset(tokenized_inputs)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Total Samples: {len(test_dataset)}")

    # 2. 추론 (Inference)
    print("Running Inference...")
    
    all_probs = [] # 클래스 1일 확률을 저장할 리스트

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 확률 계산 (Softmax)
            probs = F.softmax(logits, dim=1)
            
            # Class 1 (타겟 클래스)에 대한 확률만 추출하여 저장
            # ROC-AUC 계산을 위해서는 Label(0,1)이 아닌 이 확률값(float)이 필요함
            target_probs = probs[:, 1].cpu().numpy()
            all_probs.extend(target_probs)

    # 3. Submission 파일 생성
    submission = pd.DataFrame({
        'id': test_df[id_col],
        'target': all_probs # 0~1 사이의 실수값(확률) 저장
    })
    
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {SUBMISSION_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()