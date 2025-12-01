import os
import gc
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import rankdata # [추가됨] Rank Averaging을 위한 라이브러리
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig  # [중요] LoRA 로딩을 위한 필수 라이브러리

# ==========================================
# [1] 설정 및 하이퍼파라미터
# ==========================================
TEST_FILE_PATH = 'make-model/data/raw/test.csv'   # 실제 테스트 파일 경로 확인 필요
MODEL_ROOT_DIR = 'make-model/model/result_models/koelectra_small_lora'
OUTPUT_SUBMISSION_PATH = 'make-model/temp_submission/submission_koelectra_rank.csv'

N_FOLDS = 4
BATCH_SIZE = 64
MAX_LEN = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"⚙️ 설정 확인:")
print(f"   - Input: {TEST_FILE_PATH}")
print(f"   - Model Root: {MODEL_ROOT_DIR}")
print(f"   - Device: {DEVICE}")

# 저장 폴더가 없으면 생성
output_dir = os.path.dirname(OUTPUT_SUBMISSION_PATH)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
# [3] 추론 함수 (LoRA 적용 버전)
# ==========================================
def inference(model_path, test_loader):
    print(f"   Derived from: {model_path}")
    
    # 1. LoRA Config 로드 (Base Model 경로 확인용)
    peft_config = PeftConfig.from_pretrained(model_path)
    
    # 2. Base Model (KoELECTRA) 로드
    # config에 저장된 base_model_name_or_path를 사용하여 자동으로 원본 모델을 불러옵니다.
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path, 
        num_labels=2
    )
    
    # 3. LoRA 어댑터 결합 (Base + Adapter)
    model = PeftModel.from_pretrained(base_model, model_path)
    
    model.to(DEVICE)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="   Predicting"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Logits 추출
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0]
            
            # Softmax: Logits -> Probability (확률값 변환)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions.append(probs.cpu().numpy())
            
    # 메모리 정리
    del model, base_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return np.concatenate(predictions, axis=0)

# ==========================================
# [4] 메인 실행 (Rank Averaging 통합)
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
    input_id_col = 'id'
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
        print("⚠️ 로컬 토크나이저 로드 실패. HuggingFace Hub에서 로드 시도.")
        base_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")

    test_dataset = TestDataset(test_df, base_tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. 4-Fold 앙상블 및 예측값 수집
    
    # [중요] 각 Fold의 예측값(Class 1 확률)을 저장할 딕셔너리
    fold_predictions = {} 
    success_folds = 0

    for fold in range(N_FOLDS):
        fold_model_dir = os.path.join(MODEL_ROOT_DIR, f"fold{fold}")
        
        if not os.path.exists(fold_model_dir):
             print(f"⚠️ [Skip] 모델 폴더 없음: {fold_model_dir}")
             continue
             
        print(f"\n🔄 [Fold {fold}] Inference Start...")
        try:
            # inference 함수는 위에서 정의한 LoRA 버전 사용
            fold_probs = inference(fold_model_dir, test_loader)
            
            # 딕셔너리에 저장
            # fold_probs[:, 1]은 'AI가 썼을 확률'입니다.
            fold_predictions[f'Fold_{fold}'] = fold_probs[:, 1]
            success_folds += 1
            
        except Exception as e:
            print(f"❌ [Error] Fold {fold} 추론 실패: {e}")
            import traceback
            traceback.print_exc()

    if success_folds == 0:
        print("❌ 모든 Fold 추론 실패.")
        return

    # ==========================================
    # 4. 상관관계(Correlation) 분석 출력
    # ==========================================
    if success_folds > 1:
        print("\n" + "="*40)
        print(" 📊 Fold 간 상관관계 분석 (Correlation Matrix)")
        print("="*40)
        
        corr_df = pd.DataFrame(fold_predictions)
        correlation_matrix = corr_df.corr()
        
        print(correlation_matrix)
        print("-" * 40)
        
        min_corr = correlation_matrix.min().min()
        print(f"👉 최소 상관계수: {min_corr:.4f}")
        
        if min_corr < 0.8:
            print("🚨 모델 간 의견 차이가 큽니다. Rank Averaging이 필수적입니다.")
        else:
            print("✅ 모델들이 유사합니다. Rank Averaging을 써도 좋고 단순 평균도 좋습니다.")
        print("="*40 + "\n")

    # ==========================================
    # [핵심] 5. 결과 저장 (Rank Averaging 적용)
    # ==========================================
    print("⚖️ 최종 앙상블: Rank Averaging 적용 중...")
    
    final_rank = np.zeros(len(test_df))
    
    for fold_name, preds in fold_predictions.items():
        # 등수 매기기 (작은 값이 1등) -> 0~1 사이로 정규화
        # preds(확률)가 높을수록 높은 등수(큰 값)를 가져야 하므로 rankdata 그대로 사용
        # rankdata는 작은 값에 낮은 순위(1), 큰 값에 높은 순위(N)를 줍니다.
        # AI 확률이 높으면 -> Rank 값이 커짐 -> 최종 점수가 커짐 (맞음)
        normalized_ranks = (rankdata(preds) - 1) / (len(preds) - 1)
        final_rank += normalized_ranks
        
    # 등수 평균 계산
    avg_rank = final_rank / success_folds

    submission = pd.DataFrame()
    if input_id_col in test_df.columns:
        submission[input_id_col] = test_df[input_id_col]
    else:
        submission['id'] = test_df.index

    # 확률 대신 '정규화된 순위 평균'을 제출
    submission['generated'] = avg_rank

    submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
    print(f"✅ Rank Averaging Submission Saved: {OUTPUT_SUBMISSION_PATH}")
    print(submission.head())

if __name__ == "__main__":
    main()