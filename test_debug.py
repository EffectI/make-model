import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# ==========================================
# [1] 설정
# ==========================================
BASE_DIR = 'make-model/data/interim/'
INPUT_FILENAME = 'clean_train_balanced.csv'
OUTPUT_FILENAME = 'clean_train_balanced_under4000.csv' # 저장할 파일명
MODEL_NAME = "monologg/koelectra-small-v3-discriminator"
MAX_TOKEN_LIMIT = 4000  # 제거 기준 (이 값보다 크면 삭제)

def main():
    print("데이터 길이 필터링(Outlier Removal) 시작...")
    
    # 1. 파일 로드
    file_path = os.path.join(BASE_DIR, INPUT_FILENAME)
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    original_len = len(df)
    print(f" -> 원본 데이터 개수: {original_len:,} 개")

    # 2. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 3. 토큰 길이 계산 (progress_apply 사용)
    # tqdm을 pandas에 적용
    tqdm.pandas()
    
    print(f" -> 토큰 길이 계산 중... (기준: {MAX_TOKEN_LIMIT} 토큰)")
    
    # text 컬럼을 문자열로 변환 후 토큰 길이 계산
    df['token_len'] = df['text'].astype(str).progress_apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=True))
    )
    
    # 4. 필터링 (4000토큰 이하만 남김)
    df_filtered = df[df['token_len'] <= MAX_TOKEN_LIMIT].copy()
    
    # 제거된 개수 확인
    removed_count = original_len - len(df_filtered)
    
    # 불필요한 token_len 컬럼 제거 (선택사항)
    # df_filtered.drop(columns=['token_len'], inplace=True)
    
    # 5. 저장
    save_path = os.path.join(BASE_DIR, OUTPUT_FILENAME)
    df_filtered.to_csv(save_path, index=False)
    
    print("\n" + "="*40)
    print(" 전처리 완료")
    print("="*40)
    print(f" 삭제된 행 개수  : {removed_count:,} 개 (-{removed_count/original_len*100:.2f}%)")
    print(f" 남은 데이터 개수 : {len(df_filtered):,} 개")
    print(f" 저장 위치       : {save_path}")
    print("="*40)

if __name__ == "__main__":
    main()