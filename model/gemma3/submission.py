import os
import gc
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# [1] 설정
# ==========================================
TEST_FILE_PATH = '/home/user/rocm_project/make-model/data/raw/test.csv'
MODEL_ROOT_DIR = '/home/user/rocm_project/make-model/model/result_models/koelectra_small'
OUTPUT_SUBMISSION_PATH = './submission_ensemble.csv'

N_FOLDS = 4
BATCH_SIZE = 64
MAX_LEN = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"⚙️ 설정 확인:")
print(f"   - Input: {TEST_FILE_PATH}")
print(f"   - Model Root: {MODEL_ROOT_DIR}")
print(f"   - Device: {DEVICE}")

# ==========================================
# [2] 데이터셋 클래스
# ==========================================
class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df['text'].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }

# ==========================================
# [3] 추론 함수 (수정됨: Tuple/Logits 호환)
# ==========================================
def inference(model_path, test_loader):
    print(f"   Derived from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="   Predicting"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # -----------------------------------------------------------
            # [Fix] 출력이 Tuple인 경우와 Object인 경우 모두 처리
            # -----------------------------------------------------------
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0]  # Tuple인 경우 첫 번째 요소가 logits
            
            # Softmax: Logits -> Probability
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions.append(probs.cpu().numpy())
            
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return np.concatenate(predictions, axis=0)

# ==========================================
# [4] 메인 실행
# ==========================================
def main():
    # 1. 데이터 로드
    if not os.path.exists(TEST_FILE_PATH):
        print(f"❌ 파일이 없습니다: {TEST_FILE_PATH}")
        return

    test_df = pd.read_csv(TEST_FILE_PATH)
    print(f"📂 Test Data Loaded: {len(test_df)} rows")

    # (1) Text 컬럼 찾기
    text_col = None
    candidates = ['text', 'paragraph_text', 'content', 'sentence', 'full_text', 'overview']
    for col in test_df.columns:
        if col.lower() in candidates:
            text_col = col
            break
    
    if text_col:
        test_df.rename(columns={text_col: 'text'}, inplace=True)
    else:
        obj_cols = test_df.select_dtypes(include=['object']).columns
        if len(obj_cols) > 0:
            test_df.rename(columns={obj_cols[0]: 'text'}, inplace=True)
        else:
            print("❌ [Critical] 텍스트 컬럼을 찾을 수 없습니다.")
            return

    # (2) ID 컬럼 찾기
    input_id_col = None
    id_candidates = ['id', 'ID', 'idx', 'index', 'no']
    for col in test_df.columns:
        if col in id_candidates: 
            input_id_col = col
            break
    
    # 2. 토크나이저 준비
    first_fold_path = os.path.join(MODEL_ROOT_DIR, "fold0")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(first_fold_path)
    except:
        print("⚠️ 로컬 토크나이저 로드 실패. HuggingFace Hub(KoElectra)에서 로드 시도.")
        base_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")

    test_dataset = TestDataset(test_df, base_tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. 4-Fold 앙상블
    final_probs = np.zeros((len(test_df), 2))
    success_folds = 0

    for fold in range(N_FOLDS):
        fold_model_dir = os.path.join(MODEL_ROOT_DIR, f"fold{fold}")
        
        if not os.path.exists(fold_model_dir):
             print(f"⚠️ [Skip] 모델 폴더 없음: {fold_model_dir}")
             continue
             
        print(f"\n🔄 [Fold {fold}] Inference Start...")
        try:
            fold_probs = inference(fold_model_dir, test_loader)
            final_probs += fold_probs
            success_folds += 1
        except Exception as e:
            print(f"❌ [Error] Fold {fold} 추론 실패: {e}")

    if success_folds == 0:
        print("❌ 모든 Fold 추론 실패.")
        return

    # 4. 결과 저장
    avg_probs = final_probs / success_folds
    final_preds = np.argmax(avg_probs, axis=1)

    submission = pd.DataFrame()
    
    if input_id_col:
        submission['ID'] = test_df[input_id_col]
    else:
        submission['ID'] = test_df.index

    submission['generated'] = final_preds

    submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
    print(f"\n✅ Submission Saved: {OUTPUT_SUBMISSION_PATH}")
    print(submission.head())

if __name__ == "__main__":
    main()