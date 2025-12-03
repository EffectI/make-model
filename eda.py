# 1. 데이터 구조 및 기본 정보 확인 

import pandas as pd
import pandas as pd

train = pd.read_csv(r"C:\Users\LG\OneDrive\바탕 화면\구\4학년\경진대회\DACON\AI\train.csv")
test = pd.read_csv(r"C:\Users\LG\OneDrive\바탕 화면\구\4학년\경진대회\DACON\AI\test.csv")

train.info()
train.describe(include='all')
train.isnull().mean()

# 2. 라벨 불균형 분석 

train['generated'].value_counts(normalize=True).plot(kind='bar')

# # 3. 텍스트 길이 기반 특성 분석 

import matplotlib.pyplot as plt

train['text_len'] = train['full_text'].str.len()
train['word_count'] = train['full_text'].str.split().str.len()

print(train[['generated', 'text_len']].groupby('generated').describe())
print(train.groupby('generated')['text_len'].agg(['mean','median','std','min','max']))

fig, axes = plt.subplots(1,2, figsize=(12,5))
train.boxplot(column='text_len', by='generated', ax=axes[0])
train.boxplot(column='word_count', by='generated', ax=axes[1])
# 4. 특수문자/패턴 분석 

# 4.1 패턴 카운트 생성하기 
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind

# 한글 폰트 / 스타일 세팅은 필요하면 여기서 추가

patterns = {
    'chinese_char': r'[\u4E00-\u9FFF]',                  # 한자
    'html_tag': r'<[^>]+>',                              # HTML 태그
    'empty_parens': r'\(\s*[^\w가-힣]*\s*\)',            # 내용 없는 괄호
    'q_mark_pattern': r'\([^\(\)]{0,20}[\?\~]{1,3}[^\(\)]{0,20}\)',  # 물음표/물결이 포함된 괄호
    'repeated_dots': r'[.,]{3,}',                        # ... 같은 반복 마침표/콤마
    'repeated_parens': r'[()]{2,}',                      # 연속 괄호
    'repeated_commas': r',\s*,+'                         # 연속 쉼표
}

# 컬럼 이름 리스트 (pandas 에서 dict_keys 직접 못 쓰므로 미리 변환)
pattern_cols = list(patterns.keys())

# 패턴 count 열 생성
for name, pattern in patterns.items():
    train[name] = train['full_text'].str.count(pattern)

# ---------------------------------------------------------
# 4.2 클래스별 평균 비교 (Bar + Errorbar)
# ---------------------------------------------------------

pattern_means = train.groupby('generated')[pattern_cols].mean().T
pattern_stds  = train.groupby('generated')[pattern_cols].std().T

plt.figure(figsize=(10, 5))
x = np.arange(len(pattern_cols))
width = 0.35

plt.bar(x - width/2, pattern_means[0], width, yerr=pattern_stds[0],
        label='Human (0)', alpha=0.8)
plt.bar(x + width/2, pattern_means[1], width, yerr=pattern_stds[1],
        label='AI (1)', alpha=0.8)

plt.xticks(x, pattern_cols, rotation=45, ha='right')
plt.title("Pattern Count Comparison (Human vs AI)")
plt.ylabel("Mean Count")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4.3 패턴 Density Heatmap (논문 heatmap 스타일)
# ---------------------------------------------------------

sample_size = 500  # 너무 크니 일부 샘플링
subset = train.sample(sample_size, random_state=42)[pattern_cols]

plt.figure(figsize=(10, 6))
sns.heatmap(subset, cmap="coolwarm", cbar=True)
plt.title("Pattern Density Heatmap (500 random samples)")
plt.xlabel("Patterns")
plt.ylabel("Samples")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4.4 Logistic Regression Importance Heatmap
# ---------------------------------------------------------

X = train[pattern_cols]
y = train['generated']

# count feature라 scale 크기가 작아서 표준화는 생략해도 무방
lr = LogisticRegression(max_iter=1000, solver='liblinear')
lr.fit(X, y)
coef = lr.coef_[0]

plt.figure(figsize=(8, 2))
sns.heatmap([coef],
            annot=True,
            fmt=".3f",
            xticklabels=pattern_cols,
            cmap="coolwarm",
            center=0)
plt.title("Pattern Feature Importance (Logistic Regression)")
plt.yticks([])  # y축 눈금 제거
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4.5 통계적 유의성 검정 + 효과크기 시각화 (Cohen's d plot)
# ---------------------------------------------------------

effect_sizes = {}

for col in pattern_cols:
    x0 = train[train['generated'] == 0][col]
    x1 = train[train['generated'] == 1][col]

    # Welch t-test (p-value가 필요하면 여기서 사용)
    t_stat, p_val = ttest_ind(x0, x1, equal_var=False)

    # Cohen's d
    n0, n1 = len(x0), len(x1)
    pooled_std = np.sqrt(((n0-1)*x0.std()**2 + (n1-1)*x1.std()**2) / (n0+n1-2))
    d = (x0.mean() - x1.mean()) / pooled_std

    effect_sizes[col] = d  # 필요하면 p_val도 같이 dict에 저장 가능

plt.figure(figsize=(10, 4))
sns.barplot(x=list(effect_sizes.keys()), y=list(effect_sizes.values()), palette="crest")
plt.axhline(0.2,  color='r', linestyle='--', label='Small effect (±0.2)')
plt.axhline(-0.2, color='r', linestyle='--')
plt.title("Effect Size (Cohen's d) for Pattern Features")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()


# # 5.언어모델 기반 Metric: Perplexity & Entropy 
# #-> 금표님 시각화 자료 3개로 퉁

# # 6. Length / n-gram 고급 분석
# #6-1. 텍스트 길이 분석 강화


# # 길이 분포 Histogram + KDE
# sns.histplot(data=train, x="text_len", hue="generated", bins=100, log_scale=True)
# plt.title("Text Length Distribution (log scale)")
# plt.show()

# # 6-2. 단어 다양성 (Lexical Diversity) 비교

# #(A) Type-Token Ratio (TTR)
# def ttr(text):
#     tokens = text.split()
#     return len(set(tokens)) / (len(tokens) + 1)

# train["ttr"] = train["full_text"].sample(5000).apply(ttr)  # 일부 샘플만 계산

# #(B) 시각화
# sns.boxplot(data=train, x="generated", y="ttr")
# plt.title("Type-Token Ratio by Class")
# plt.show()

# #6-3. Char n-gram 빈도 기반 분석 (가장 중요한 파트)

# # (A) TF-IDF로 char 3·4·5gram 벡터화
# from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf_char = TfidfVectorizer(
#     analyzer='char',
#     ngram_range=(3,5),
#     max_features=20000
# )

# X_char = tfidf_char.fit_transform(train["full_text"].sample(20000))

# #(B) n-gram feature와 라벨의 상관도(점수) 계산

# from sklearn.feature_selection import mutual_info_classif

# mi = mutual_info_classif(X_char, train["generated"].sample(20000), discrete_features=True)

# #(C) 상위 20개 n-gram을 뽑아 시각화
# import numpy as np

# top_idx = np.argsort(mi)[-20:]
# top_scores = mi[top_idx]
# top_features = np.array(tfsidf_char.get_feature_names_out())[top_idx]

# sns.barplot(x=top_scores, y=top_features)
# plt.title("Top 20 Informative Char n-grams")
# plt.show()

# =========================
# 공통 준비 (이미 했다면 생략 가능)
# =========================
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# 텍스트 길이 / 단어 수 컬럼 (이미 있으면 건너뜀)
if 'text_len' not in train.columns:
    train['text_len'] = train['full_text'].str.len()

if 'word_count' not in train.columns:
    train['word_count'] = train['full_text'].str.split().str.len()

# generated: 0=Human, 1=AI
label_map = {0: 'Human', 1: 'AI'}
train['label_name'] = train['generated'].map(label_map)

# =========================================
# 6-1. 텍스트 길이 분석 강화 (Histogram + KDE)
# =========================================

plt.figure(figsize=(8, 5))
sns.histplot(
    data=train,
    x='text_len',
    hue='label_name',
    bins=80,
    log_scale=True,        # x축 log
    kde=True,
    element='step',
    stat='count',
    common_norm=False
)
plt.title("Text Length Distribution (log scale, with KDE)")
plt.xlabel("Text length (characters, log scale)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 길이 분포 요약 테이블 (이미 있는 내용이지만 코드로 다시 생성)
length_summary = (
    train.groupby('generated')['text_len']
         .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
)
print("\n[Text Length Summary by Label]")
print(length_summary)

# ===================================
# 6-2. 단어 다양성 (Lexical Diversity, TTR)
# ===================================

def compute_ttr(text: str) -> float:
    """
    Type-Token Ratio: unique_token / total_token
    너무 짧은 문서는 0으로 처리
    """
    if not isinstance(text, str):
        return 0.0
    # 단어 토큰 (영문/숫자/한글 포함)
    tokens = re.findall(r'\w+', text)
    n = len(tokens)
    if n == 0:
        return 0.0
    return len(set(tokens)) / n

# TTR 계산 (시간이 좀 걸릴 수 있음)
if 'ttr' not in train.columns:
    train['ttr'] = train['full_text'].apply(compute_ttr)

# TTR 통계 요약
ttr_summary = (
    train.groupby('generated')['ttr']
         .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
         .rename(index=label_map)
)
print("\n[Type-Token Ratio (TTR) Summary by Label]")
print(ttr_summary)

# TTR Boxplot
plt.figure(figsize=(5, 5))
sns.boxplot(data=train, x='label_name', y='ttr')
plt.title("Type-Token Ratio by Class")
plt.xlabel("Label")
plt.ylabel("TTR")
plt.tight_layout()
plt.show()

# ==========================
# 7. 문체 분석 (Stylistic Features)
# ==========================

def count_sentences(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    # 문장 끝을 기준으로 rough split
    sentences = re.split(r'[\.!?…\n]+', text)
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)

def avg_sentence_len_tokens(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    sentences = re.split(r'[\.!?…\n]+', text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    lens = [len(re.findall(r'\w+', s)) for s in sentences]
    return np.mean(lens)

def punctuation_ratio(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    punct = re.findall(r'[.,;:!?]', text)
    return len(punct) / len(text)

def digit_ratio(text: str) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    digits = sum(ch.isdigit() for ch in text)
    return digits / len(text)

# 문체 feature 계산
sty_cols = ['sentence_count', 'avg_sent_len', 'punct_ratio', 'digit_ratio']

if not set(sty_cols).issubset(train.columns):
    print("\n[Computing stylistic features ...]")
    train['sentence_count'] = train['full_text'].apply(count_sentences)
    train['avg_sent_len']   = train['full_text'].apply(avg_sentence_len_tokens)
    train['punct_ratio']    = train['full_text'].apply(punctuation_ratio)
    train['digit_ratio']    = train['full_text'].apply(digit_ratio)

# 라벨별 평균 비교 테이블
sty_summary = (
    train.groupby('generated')[sty_cols]
         .mean()
         .rename(index=label_map)
)
print("\n[Stylistic Feature Means by Label]")
print(sty_summary)

# Heatmap (라벨별 문체 차이)
plt.figure(figsize=(6, 3))
sns.heatmap(sty_summary.T, annot=True, fmt=".3f", cmap="Blues")
plt.title("Stylistic Feature Means (Human vs AI)")
plt.ylabel("Feature")
plt.xlabel("Label")
plt.tight_layout()
plt.show()

# 개별 boxplot (예: 평균 문장 길이, 문장 수)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(data=train, x='label_name', y='avg_sent_len', ax=axes[0])
axes[0].set_title("Average Sentence Length (tokens)")
axes[0].set_xlabel("Label")
axes[0].set_ylabel("Avg sentence length")

sns.boxplot(data=train, x='label_name', y='sentence_count', ax=axes[1])
axes[1].set_title("Sentence Count per Document")
axes[1].set_xlabel("Label")
axes[1].set_ylabel("Sentence count")

plt.tight_layout()
plt.show()

# =========================================
# 8. 텍스트 유사도 기반 이상치 / 중복 탐지
#    (Sentence-BERT + Cosine Similarity)
# =========================================

# ★주의★: 전체 97k를 다 쓰면 메모리/시간 과다 → 샘플링 사용
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("\n[INFO] sentence-transformers 미설치 →")
    print("       pip install sentence-transformers 로 설치 후 아래 코드 재실행 필요")
else:
    # 1) 샘플링 (예: 3,000개)
    sample_size = 3000
    subset = (
        train[['full_text', 'generated']]
        .sample(sample_size, random_state=42)
        .reset_index(drop=True)
    )
    print(f"\n[Similarity EDA] Using subset of size {len(subset)}")

    # 2) 멀티링구얼 SBERT 로딩 (한국어 지원 모델)
    model_name = "distiluse-base-multilingual-cased-v1"
    model = SentenceTransformer(model_name)

    # 3) 임베딩 계산
    embeddings = model.encode(
        subset['full_text'].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # 4) 코사인 유사도 행렬 (n x n)
    # 메모리 고려해서 float32로 변환
    embeddings = embeddings.astype('float32')
    sim_matrix = cosine_similarity(embeddings)

    # 자기 자신(대각선)은 제외하기 위해 -1로 세팅
    np.fill_diagonal(sim_matrix, -1.0)

    # 5) 각 샘플별 최대 유사도 및 인덱스
    max_sim = sim_matrix.max(axis=1)
    max_idx = sim_matrix.argmax(axis=1)

    subset['max_sim'] = max_sim
    subset['max_sim_idx'] = max_idx

    # 6) threshold 이상을 "중복/유사 문서"로 간주
    threshold = 0.97
    dup_mask = subset['max_sim'] >= threshold
    dup_subset = subset[dup_mask]

    print(f"\n[High-Similarity Pairs (sim >= {threshold})]")
    print(f"Count: {len(dup_subset)} / {len(subset)}")

    # 상위 10개 pair만 테이블로 보기
    top_n = 10
    pairs = []
    for i, row in dup_subset.head(top_n).iterrows():
        j = int(row['max_sim_idx'])
        pairs.append({
            'idx_i': i,
            'idx_j': j,
            'sim': row['max_sim'],
            'label_i': label_map[subset.loc[i, 'generated']],
            'label_j': label_map[subset.loc[j, 'generated']],
            'text_i_snippet': subset.loc[i, 'full_text'][:80].replace('\n', ' ') + "...",
            'text_j_snippet': subset.loc[j, 'full_text'][:80].replace('\n', ' ') + "..."
        })

    dup_pairs_df = pd.DataFrame(pairs)
    pd.set_option('display.max_colwidth', 120)
    print("\n[Top High-Similarity Pairs]")
    print(dup_pairs_df)

    # 7) 유사도 높음 문서 수를 라벨별로 집계
    dup_counts = dup_subset['generated'].map(label_map).value_counts()

    plt.figure(figsize=(4, 4))
    sns.barplot(x=dup_counts.index, y=dup_counts.values)
    plt.title(f"Highly-Similar Documents by Label (sim >= {threshold})")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# =========================================
# 9. 최종 EDA 요약용 핵심 수치만 정리 (텍스트 출력)
# =========================================

print("\n================= EDA FINAL SUMMARY (핵심 포인트) =================")
print(f"- Label imbalance: Human 0 → { (train['generated']==0).mean():.3% }, "
      f"AI 1 → { (train['generated']==1).mean():.3% }")

print("- Text length: Human mean {:.1f}, AI mean {:.1f} (Human slightly longer, "
      "variance larger → extreme long docs 존재)".format(
          length_summary.loc[0, 'mean'], length_summary.loc[1, 'mean']))

print("- TTR: Human mean {:.3f}, AI mean {:.3f} (Human이 어휘 다양성 더 높음)".format(
    ttr_summary.loc['Human', 'mean'], ttr_summary.loc['AI', 'mean']))

print("- Stylistic: 문장 수 / 평균 문장 길이 / 구두점 비율에서 Human과 AI 간 미묘한 차이 존재 "
      "(표/heatmap 참고)")

print("- Special char patterns: 평균 카운트는 존재하지만 label 구분력은 거의 없음 "
      "→ 필요 시 노이즈 제거 수준으로만 사용")

print("- Perplexity & Entropy: 두 지표 간 상관관계 높음 (R≈0.9), Human vs AI 분포 차이 제한적 "
      "→ 단독 feature로는 효과 낮음")
print("===================================================================")
