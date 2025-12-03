![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow) ![License](https://img.shields.io/badge/License-MIT-green)
````markdown
# 🤖 AI-Text-Classifier (Lightweight Model Project)

EDA

# 1. 데이터 기본 구조
   
python eda.py
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 97172 entries, 0 to 97171
Data columns (total 3 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   title      97172 non-null  object
 1   full_text  97172 non-null  object
 2   generated  97172 non-null  int64
dtypes: int64(1), object(2)
memory usage: 2.2+ MB

- 전체 97,172개 문서로 구성
- generated: 0(Human) / 1(AI)
- 레이블 비율: 89,177(0) : 7,995(1) ≈ 11:1의 강한 불균형 존재

→ 이후 모델 학습 시 downsampling 또는 re-weighting 필요함.

# 2. Special Character Pattern Analysis (특수문자 패턴 분석)

사용된 패턴:
chinese_char, html_tag, empty_parens, qmark_pattern, repeated_dots, repeated_parens, repeated_commas

<img width="1000" height="925" alt="boxblot grouped by generated" src="https://github.com/user-attachments/assets/5e881d69-a0f7-4e5e-bd56-b9df3a11fd17" />
<img width="500" height="400" alt="boxplot grouped by generated empty parens (log scale)" src="https://github.com/user-attachments/assets/bb9a4ae2-2f11-4cb0-a282-185e82c068e6" />
<img width="500" height="400" alt="boxplot grouped by generated qmark paterns (log scale)" src="https://github.com/user-attachments/assets/70d81c06-690d-4dc1-bb79-d9c4f58356d6" />
<img width="500" height="400" alt="boxplot grouped by generated repeated commas (log scale)" src="https://github.com/user-attachments/assets/12159376-750c-4688-8594-b9d55cccb123" />
<img width="500" height="400" alt="boxplot grouped by generated repeated dots (log scale)" src="https://github.com/user-attachments/assets/82535283-a7b1-423f-bb43-b2cff97d66f4" />
<img width="500" height="400" alt="boxplot grouped by generated repeated parens (log scale)" src="https://github.com/user-attachments/assets/4c1bd1e6-f3f0-467a-9cb8-f52e6a8cba84" />
<img width="999" height="399" alt="Effect Size (Cohen&#39;s d) for Pattern Features" src="https://github.com/user-attachments/assets/7c098e28-acd0-4191-865e-5a4cedef1acb" />
<img width="1489" height="988" alt="Special Character Pattern Comparison (Human vs AI" src="https://github.com/user-attachments/assets/dcc36df9-5eb0-44e4-b0a9-4e7ea33db5e3" />
<img width="599" height="299" alt="Stylistic Feature Means (Human vs AI)" src="https://github.com/user-attachments/assets/23b13c0f-a79a-4a11-9e0b-920879381bac" />

한자, HTML Tag, 반복 괄호/마침표/쉼표 등 총 7개 패턴을 분석한 결과,
Human–AI 간 차이는 존재하나 절대적 크기가 매우 작고 구분력이 낮음을 확인함.

분석 결과 요약

| Pattern         | Human | AI    | 차이    |
| --------------- | ----- | ----- | -----   |
| chinese_char    | 11.7  | 6.5   | 약 5개  |
| html_tag        | 0.125 | 0.099 | 0.02개  |
| repeated_parens | 0.031 | 0.017 | 0.01개  |

결론
- 모든 패턴의 절대값이 너무 작음 (거의 0~0.1대).
- Cohen’s d 효과크기 역시 전부 0.1 이하(= 매우 작은 효과).
 → 구분력 극히 낮음
 → Feature로 사용 시 모델 성능에 기여 불가
 → 전처리(특수문자 제거) 단계에서만 사용하고 Feature에서는 제외함


# 3. 언어모델 기반 Metric: Perplexity & Entropy

사용 이미지들
<img width="1184" height="648" alt="Entrpy Distribution" src="https://github.com/user-attachments/assets/abe8ca5b-7f77-4fab-9212-5ca0630d4ccc" />
<img width="1184" height="648" alt="Perplexity Distribution (Log Scale)" src="https://github.com/user-attachments/assets/2f3aa5e7-74fd-416a-85f6-e11352b26670" />
<img width="1184" height="816" alt="Perplexity vs Entropy" src="https://github.com/user-attachments/assets/39a172de-63be-4131-8a7c-7b2ff482972e" />

분석 결과

Human 문서
- Perplexity 분포 폭이 넓고 평균이 높음
- LM이 예측하기 어려운 문장 구조 → perplexity↑ / entropy↑

AI 문서
- 예측 패턴이 정형적 → perplexity↓, entropy↓
- 분포가 좁고 안정적

핵심 결론
- Perplexity는 Human–AI를 나누는 가장 강력한 Feature
- 길이·특수문자 기반 Feature보다 구분 성능(AUC) 기여도가 압도적

# 4. 텍스트 길이 분석 (Length Distribution + KDE)

<img width="1200" height="500" alt="boxplot text_len word_count" src="https://github.com/user-attachments/assets/3e1485fe-84d2-44dd-91fb-d54d514dbebf" />
<img width="640" height="480" alt="length vs word count scatter" src="https://github.com/user-attachments/assets/582d5903-49a6-4141-accc-a4c64d1597cd" />
<img width="640" height="480" alt="text length distribution (log scale)" src="https://github.com/user-attachments/assets/5b8fb3e7-23e7-4339-b1a2-1199dfbcae50" />
<img width="799" height="499" alt="Text Length Distribution (log scale, with KDE)" src="https://github.com/user-attachments/assets/282d32de-9d6d-4ff4-95ae-2c81d22a31ef" />
<img width="1184" height="648" alt="Text Length Statistics by Label" src="https://github.com/user-attachments/assets/ac754680-743b-46fd-8821-81d21f267c21" />
<img width="1222" height="648" alt="Text Length Summary (by Label)" src="https://github.com/user-attachments/assets/4b3ae0ac-8362-4044-941c-865ba7c964ef" />

정량 분석
| Metric | Human  | AI     | Insight              |
| ------ | ------ | ------ | ---------------      |
| Mean   | 2325   | 2298   | Human이 약간 길다     |
| Median | 1331   | 1334   | 거의 동일             |
| Std    | 3351   | 3131   | Human 변동성↑         |
| Max    | 98,549 | 46,814 | Human 극단적 장문 존재 |


Histogram + KDE 결과 Human과 AI 문서 길이 분포에 뚜렷한 차이가 확인됨.

Human 텍스트
- 매우 짧은 글부터 매우 긴 글까지 범위 전반에 분포
- 문서 길이의 불규칙성·다양성↑

AI 텍스트
- 특정 길이 구간(예: 300~1,000 tokens)에 밀집
- 생성 모델의 출력 길이가 일정한 규칙성을 가짐

→ 문서 길이만으로도 weak classifier 역할 가능.

결론
- Human 문서는 매우 짧은 글~수 만 단어까지 폭넓게 분포
- AI 문서는 300~1,000 tokens 부근에 집중되는 경향
→ 문서 길이만으로도 약한 분류기(weak classifier) 역할 가능


# 5. Lexical Diversity (어휘 다양성, TTR: Type–Token Ratio)

사용 이미지
<img width="640" height="480" alt="lexical diversity _ type-toke ratio" src="https://github.com/user-attachments/assets/8703ea2f-cc6a-429e-a528-9808d7183f4d" />
<img width="499" height="499" alt="Type-Token Ratio by Class" src="https://github.com/user-attachments/assets/c66f2cfc-e3d0-4b8c-bc3d-51ab8223552c" />

분석 결과

Human
- 다양한 표현·단어 사용
- 문장 구조·주제 전환이 자유로워 어휘 분포 폭이 넓음
- TTR 높음
→ 다양한 표현·단어 조합 → TTR 높음

AI
- 동일 표현 반복
- 문장 패턴의 규칙성
- TTR 낮음
→ AI: 반복적 표현·정형 패턴 → TTR 낮음

→ 문체적 변동성 자체가 Human의 중요한 시그널로 작동함

# 6. Stylistic Analysis (문체 분석)

사용 이미지 
<img width="599" height="299" alt="Stylistic Feature Means (Human vs AI)" src="https://github.com/user-attachments/assets/951f7936-e99d-42c5-b2e5-7328f27f9150" />

문장 길이 변화, 문장부호 패턴, 불용어 비율, 구조적 다양성을 정량화함.

Human 문체
- 문장 길이의 분산이 큼
- 쉼표/콜론/대시 등 다양한 부호 사용
- 주제 전환과 흐름 변화가 빈번
→ 자연스러운 문체적 변동성

AI 문체
- 문장 길이가 규칙적인 패턴으로 정렬
- ‘하지만’, ‘따라서’, ‘또한’ 등 연결어 반복
- 부호 사용 방식이 일정
→ 기계적 일관성

→ 문체 분석은 의미 있는 보조 Feature로 활용 가능.

# 7. 텍스트 유사도 기반 이상치·중복 탐지 (TF-IDF Cosine Similarity)

TF-IDF 기반 문서 간 유사도를 계산한 결과:

AI 문서
- 높은 유사도 클러스터 존재
- 0.8~0.95 이상의 유사도 그룹이 다수
- LLM의 반복적 생성 패턴이 확연함

Human 문서
- 유사도 분포 폭이 넓고 다양성 존재
- 고유 문체로 인해 유사도가 낮게 분산됨

→ 유사도 분석으로 AI 문서의 반복 생성 행태를 효과적으로 탐지 가능.

# 8. EDA 종합 결론

- Human 텍스트는 길이·문체·어휘 다양성에서 높은 변동성을 보임
- AI 텍스트는 길이·문체·패턴이 일정한 규칙성을 가짐
- 특수문자 패턴 7종은 구분력 거의 없음 → Feature 제외
- Perplexity/Entropy는 가장 강력한 확률 기반 Feature
- 유사도 분석에서 AI 문서 특유의 높은 반복성이 명확히 드러남
- 최종적으로 길이·다양성·문체·LM 기반 Feature가 Human–AI 분류의 핵심 요소임



## 📖 개요 (Overview)

본 프로젝트는 **AI 생성 텍스트와 인간 작성 텍스트를 탐지 및 분류**하는 머신러닝 모델링 프로젝트입니다.
거대 언어 모델(LLM)보다는 **경량화된 모델(Lightweight Models)**을 우선적으로 활용하여 효율적인 추론 성능을 확보하는 것을 목표로 합니다.

**핵심 목표:**
* **데이터 파이프라인:** 전처리 및 4-Fold 교차 검증 데이터셋 구축
* **모델링:** `klue/roberta-small`, `koelectra-small` 등 경량 모델 기반 Fine-tuning
* **성능 극대화:** 4-Fold 앙상블(Ensemble)을 통한 일반화 성능 및 정확도 향상

---

## 📂 폴더 구조 (Project Structure)

```text
make-model/
├── data/
│   ├── raw/               # [수정 금지] 원본 데이터 (train.csv, test.csv)
│   ├── interim/           # 중간 정제 데이터 (clean_train.csv 등)
│   └── fold/              # 학습용 4-Fold 분할 데이터셋
├── model/
│   ├── result_models/     # 학습 완료된 모델 저장소 (.pth, .bin)
│   ├── Klue_roberta-small.ipynb  # 모델 학습용 노트북 A
│   ├── koelectra-small.ipynb     # 모델 학습용 노트북 B
│   └── ... 
├── Data Preprocessing.ipynb      # [Step 1] 전처리 및 데이터 분할
├── Ensemble.ipynb                # [Step 2] 최종 앙상블 및 추론
└── README.md
````

> **⚠️ 주의사항:**
> 모든 Notebook 파일은 **프로젝트 최상위(`make-model/`) 폴더를 기준**으로 경로가 설정되어 있습니다.
> 하위 폴더로 파일을 이동시키거나, 작업 경로(Current Working Directory)가 다를 경우 경로 에러가 발생할 수 있습니다.

-----

## 🚀 실행 가이드 (Workflow)

프로젝트는 아래 순서대로 실행해야 올바르게 동작합니다.

### 1️⃣ Step 1: 데이터 전처리

  * **파일:** `Data Preprocessing.ipynb`
  * **설명:** 원본 데이터(`data/raw`)의 특수문자 제거, 데이터 정제 후 `processed` 폴더에 저장하고, 4-Fold 검증용 데이터셋을 생성합니다.

### 2️⃣ Step 2: 모델 학습 (Training)

  * **파일:** `model/` 폴더 내의 각 모델별 노트북 (예: `Klue_roberta-small.ipynb`)
  * **설명:** 생성된 4-Fold 데이터를 사용하여 모델을 학습시킵니다. 학습된 모델은 `model/result_models/`에 저장됩니다.

### 3️⃣ Step 3: 앙상블 및 추론 (Inference)

  * **파일:** `Ensemble.ipynb`
  * **설명:** 각 Fold에서 학습된 모델들을 불러와 **Soft Voting** 방식으로 앙상블하여 최종 예측 결과(`submission.csv`)를 생성합니다.

-----

## 💻 시작하기 (Getting Started)

팀원 환경 세팅을 위해 아래 절차를 따라주세요.

### 1\. 환경 설정 (Installation)

필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

### 2\. 데이터 파일 배치 (Data Setup)

보안 및 용량 문제로 데이터 파일은 Git에 포함되지 않습니다.
공유받은 `train.csv`, `test.csv` 파일을 아래 경로에 위치시켜 주세요.

```text
make-model/data/raw/train.csv
make-model/data/raw/test.csv
```

-----

## 📊 실험 결과 (Experiment Results)

다양한 모델 아키텍처와 실험 조건에 따른 성능 비교표입니다.

| Model Name | Backbone | F1-Score | Accuracy | Note |
| :--- | :--- | :---: | :---: | :--- |
| **KLUE RoBERTa** | `klue/roberta-small` | 0.0000 | 0.0000 | Baseline |
| **KoELECTRA** | `monologg/koelectra-small` | 0.0000 | 0.0000 | - |
| **Ensemble (Soft)** | 4-Fold Integration | **0.0000** | **0.0000** | Best Performance |

> **Note:**
>
>   * **Metric:** Macro F1-Score 및 Accuracy 기준
>   * **Environment:** Google Colab T4 / Local GPU (RTX 3060)

```
```