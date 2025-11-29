#unset HSA_OVERRIDE_GFX_VERSION
import pandas as pd
import numpy as np
import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ==========================================
# [설정] 경로 및 모델
# ==========================================
# 이전 단계에서 저장된 파일 경로
BASE_DIR = Path('/home/user/rocm_project/make-model/model/result_models/koelectra_small')
ERROR_FILE_PATH = BASE_DIR / 'hard_cases_all_wrong.csv'

# 가장 성능이 좋았던 Fold의 모델을 불러와서 분석 (예: fold0)
MODEL_PATH = BASE_DIR / 'fold1'
MODEL_NAME = "monologg/koelectra-small-v3-discriminator"
MAX_LEN = 512

# ==========================================
# [함수] 문장 분리 및 추론
# ==========================================
def split_sentences(text):
    #? ! 뒤에 공백이 오면 자름
    # 문장 끝 부호 뒤에 공백이 있는 경우 분리
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s for s in sentences if len(s.strip()) > 5] # 너무 짧은 문장 제외

def analyze_sentences():
    # 1. 데이터 및 모델 로드
    if not ERROR_FILE_PATH.exists():
        print("오답 파일이 없습니다. 이전 코드를 먼저 실행해주세요.")
        print(f"   찾는 경로: {ERROR_FILE_PATH}")
        return

    df_wrong = pd.read_csv(ERROR_FILE_PATH)
    print(f"📂 분석 대상: {len(df_wrong)}개 (4개 모델 모두 틀린 케이스)")

    print("⏳ 모델 로딩 중...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 토크나이저는 모델 이름으로 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 모델은 저장된 경로에서 로드 (pytorch_model.bin이 있는 폴더)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()

    results = []

    # 2. 각 데이터별 문장 분석
    print("🚀 문장 단위 분석 시작 (범인 색출 중...)")
    for idx, row in tqdm(df_wrong.iterrows(), total=len(df_wrong)):
        doc_id = row['id']
        full_text = row['text']
        true_label = row['label']

        # 문장 분리
        sentences = split_sentences(full_text)

        if not sentences:
            continue

        # 문장별 추론
        inputs = tokenizer(sentences, truncation=True, padding=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # outputs.logits로 접근하는 대신 튜플의 첫 번째 요소를 사용합니다.
            logits = outputs[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs_1 = probs[:, 1].cpu().numpy() # AI일 확률

        # 가장 AI스러운 문장 찾기
        max_prob_idx = np.argmax(probs_1)
        max_prob = probs_1[max_prob_idx]
        culprit_sentence = sentences[max_prob_idx]

        results.append({
            'id': doc_id,
            'max_prob_ai': max_prob, # 이 문장이 AI일 확률
            'culprit_sentence': culprit_sentence, # 문제의 그 문장
            'full_text_preview': full_text[:50] + "..."
        })

    # 3. 결과 저장
    result_df = pd.DataFrame(results)
    # AI 확률이 높은 순서대로 정렬
    result_df = result_df.sort_values(by='max_prob_ai', ascending=False)

    save_path = BASE_DIR / 'culprit_sentences_analysis.csv'
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print(f"분석 완료 저장 경로: {save_path}")
    print("="*50)

    # 상위 3개 미리보기
    print("\n[TOP 3 sentences likely to be AI-generated]")
    for i in range(min(3, len(result_df))):
        row = result_df.iloc[i]
        print(f"\n{i+1}위 (AI 확률: {row['max_prob_ai']:.4f})")
        print(f"   문장: \"{row['culprit_sentence']}\"")
        print(f"   원본ID: {row['id']}")

if __name__ == "__main__":
    analyze_sentences()