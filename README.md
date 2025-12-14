# 🤖 AI-Text-Classifier  
### Lightweight Human–AI Text Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔑 Executive Summary

본 프로젝트는 **AI 생성 텍스트와 인간 작성 텍스트를 구분**하기 위한  
**경량화(Lightweight) 모델 기반 분류 파이프라인**을 구축한다.

EDA를 통해 **Perplexity·Entropy·문체 변동성**이  
Human–AI 구분에 있어 핵심 신호임을 실험적으로 검증하였으며,  
구분력이 낮은 Feature는 명시적으로 배제하였다.

최종적으로 **4-Fold 교차검증 + Soft Voting 앙상블**을 적용하여  
재현 가능하고 일반화 성능이 높은 분류 구조를 목표로 한다.

---

## 📌 Project Motivation & Scope

최근 LLM의 발전으로 AI 생성 텍스트는  
문법·어휘 측면에서 인간 텍스트와 거의 구분되지 않는다.  
그러나 **예측 가능성, 문체 변동성, 반복성** 관점에서는  
여전히 구조적 차이가 존재한다.

본 프로젝트는 다음 질문에서 출발한다.

```text
“대형 언어 모델 없이도,
통계적·문체적 신호만으로
AI 생성 텍스트를 식별할 수 있는가?”
라는 전략을 채택하였다.
```

이를 위해 다음 전략을 채택하였다. 
``` text
- 대형 LLM API 사용 ❌
- 경량 Transformer 기반 분류 ⭕
- EDA 기반 Feature 선택 ⭕
```
즉, 성능–효율 균형을 고려한 실전형 분류 파이프라인을 목표로 한다.

---

## 🗂 Table of Contents

1. Dataset Overview  
2. Exploratory Data Analysis (EDA) – Decision-Driven Summary  
3. Feature Evaluation & Selection (Keep vs Discard)  
4. Modeling Strategy  
5. Training & Inference Pipeline  
6. Experiments & Evaluation  
7. Reproducibility & Environment  
8. Repository Structure  
9. License  

---

## 1️⃣ Dataset Overview

본 프로젝트에서 사용한 데이터셋의 기본 구성은 다음과 같다.

```
Total Documents: 97,172

Columns:
- title (string)
- full_text (string)
- generated (0 = Human, 1 = AI)
```
### Class Distribution

```text
- Human (0): 89,177  
- AI (1): 7,995  
- Ratio ≈ **11 : 1 (Severely Imbalanced)**
```

📌 **Implication:**  
```text
Accuracy 단독 평가는 부적절하며,  
**Macro F1-Score** 기반 평가 및  
downsampling / class weighting 전략이 필요하다.
```
본 데이터 특성은 이후 EDA 및 학습 전략 설계의 핵심 전제로 활용된다.
---

## 2️⃣ Exploratory Data Analysis (EDA) – Decision-Driven Summary

EDA의 목적은 단순한 데이터 요약이 아니라,  
**Feature 후보의 생존 여부를 판단하기 위한 의사결정 근거를 확보하는 것**이다.

분석 결과, Feature들은 다음 세 그룹으로 명확히 구분되었다.

---

### ✅ Strong Signal

#### ▸ Perplexity & Entropy (Language-Model-based)

Human과 AI 텍스트 간 **언어 모델 관점의 예측 난이도 차이**가 뚜렷하게 관찰되었다.

```text
Human Text
- Perplexity 분포 폭이 넓음
- 예측 불확실성(Entropy) 높음

AI Text
- Perplexity 값이 낮음
- 분포가 매우 안정적
```

📌 **결론:**  
→ 가장 강력한 분리 신호  
→ 본 프로젝트의 핵심 Feature

---

### ⚠️ Medium Signal

#### ▸ Text Length / Lexical Diversity (TTR) / Stylistic Variance

```text
Human Text
- 문서 길이 및 문장 길이 분산 큼
- 어휘 분포 및 문체 변동성 높음

AI Text
- 특정 길이 구간에 집중
- 반복적인 연결어 및 문장 패턴
```

📌 **결론:**  
→ 단독 Feature로는 약함  
→ 다른 Feature와 결합 시 보조적 성능 향상

---

### ❌ Weak Signal (Discarded)

#### ▸ Special Character Patterns
특수문자 기반 패턴은 통계적으로 유의미한 차이를 보이지 않았다.

```text
Patterns:
- 한자
- HTML Tag
- 반복 괄호 / 마침표 / 쉼표

Observations:
- 대부분의 Feature 값이 0에 수렴
- Cohen’s d < 0.1
```
📌 **결론:**  
→ 구분력 부족
→ Feature로 사용하지 않고 전처리 단계에서만 활용

📎 모든 EDA 시각화·통계는 `notebooks/EDA.ipynb`에 보존됨.

---

## 3️⃣ Feature Evaluation & Selection (Keep vs Discard)

EDA 결과를 바탕으로,  
각 Feature를 **구분력·안정성·실제 기여도** 관점에서 평가하였다.

---

### Feature Selection Summary

| Feature Category | Decision | Rationale |
|------------------|----------|-----------|
| Perplexity / Entropy | ✔ Keep | 가장 강력한 분리 신호 |
| Text Length | ✔ Keep | Weak classifier로 활용 가능 |
| Lexical Diversity (TTR) | ✔ Keep | 문체 변동성 반영 |
| Stylistic Metrics | ✔ Keep | 구조적 반복성 탐지 |
| Special Characters | ✘ Discard | 효과크기 미미 |

---

### Selection Rationale

```text
1. Perplexity / Entropy
   - Human–AI 간 분포 차이가 가장 명확
   - 단일 Feature 기준에서도 높은 분리력 확보

2. Text Length / Lexical Diversity / Stylistic Metrics
   - 단독 Feature로는 제한적이나
   - 결합 시 문체적 변동성과 반복성 포착 가능
   - 보조 Feature로서 성능 향상에 기여

3. Special Character 기반 Feature
   - 절대값 대부분이 0에 수렴
   - Cohen’s d < 0.1
   - 통계적으로 유의미한 구분력 없음
```

📌 모든 Feature 선택은 실험 결과에 근거하여 결정되었으며,
직관이나 추측에 의존한 Feature는 포함되지 않는다.

---

## 4️⃣ Modeling Strategy

### Backbone Models (Lightweight)

본 프로젝트는 경량화 모델(Lightweight Transformer)을 중심으로  
성능–효율 균형을 맞춘 분류 파이프라인을 설계하였다.

선택한 Backbone 모델은 다음과 같다.

- `klue/roberta-small`
- `monologg/koelectra-small`

```text
Selection Rationale
- 한국어 사전학습(pretrained) 모델
- 파라미터 수 대비 표현력 우수
- 빠른 학습 및 추론 가능
- On-device / On-premise 환경 적용 가능
```

Validation Strategy

모델의 일반화 성능을 안정적으로 평가하기 위해 
4-Fold Cross Validation 전략을 채택하였다. 
```text
Validation Setup
- 동일 데이터에 대한 4-Fold 분할
- Fold별 독립 학습 수행
- Fold 간 성능 편차 분석
```

Ensemble Strategy (Soft Voting)

각 Fold에서 학습된 모델의 예측 확률을
Soft Voting 방식으로 통합하였다.

```text
Why Soft Voting Ensemble?
- 단일 모델의 편향 감소
- Fold 간 분산 완화
- 예측 안정성 향상
- Macro F1 기준 일반화 성능 개선
```

---

## 5️⃣ Training & Inference Pipeline

### Step 1. Preprocessing & Fold Generation

본 프로젝트의 데이터 전처리는 단순 정제가 아닌,  
**학습 난이도 제어(Difficulty Control)**를 목표로 설계되었다.

---

#### (1) Cleaning

다음 노이즈 패턴을 제거하였다.

- HTML Tag
- 한자
- 빈 괄호
- 반복 점 (`...`)
- 반복 쉼표 (`,,`)
- 중복 괄호
- 다중 공백

```text
Before: 97,172 documents
After : 97,041 documents
Retention Rate: 99.87%
```
→ 정보 손실 없이 노이즈만 제거

---

#### (2) Difficulty-Aware Downsampling (핵심 아이디어)

AI 문체는 특정 설명체 어미 및 논리 연결어 패턴을 반복적으로 사용한다는
가설에 기반하여 Style Score 기반 가중치 샘플링을 적용하였다.

Style Score 정의

- 설명체 어미 (다, 임, 함) → +1.0
- 논리 연결어 (따라서, 또한, 이에, 즉, 의미한다) → +0.5

Downsampling Policy

- Top 60% (설명문·논리체, Hard Negative) → 유지
- Bottom 40% (대화체·쉬운 문장, Easy Negative) → 랜덤 제거

```text
Human (0): 89,177 → 23,985
AI    (1): 7,995
Final Ratio ≈ 3 : 1

Average Style Score
Before: 0.975
After : 0.989
```
→ 모델이 헷갈릴 수밖에 없는 어려운 데이터만 남겨 효율적 학습 유도

---

### Step 2. Model Training (per Fold)

각 Fold별로 아래 노트북을 사용하여 모델을 학습한다.
- Klue_roberta-small.ipynb
- koelectra-small.ipynb

```text
Trained model checkpoints are saved to:
model/result_models/
```
---

### Step 3. Ensemble & Prediction

- 사용 노트북: Ensemble.ipynb
- 앙상블 방식: Soft Voting Ensemble

```text
Final Output:
submission.csv
```

---

## 6️⃣ Experiments & Evaluation

### Evaluation Results

| Model | Backbone | Metric | Performance |
|------|----------|--------|-------------|
| KLUE RoBERTa | roberta-small | Macro F1 | ~0.72 |
| KoELECTRA | electra-small | Macro F1 | ~0.70 |
| Ensemble | Soft Voting | Macro F1 | **≈ 0.84 (Best)** |

---

### Evaluation Metrics

- **Macro F1-Score** (Primary)
- **Accuracy** (Secondary)

---

### Result Analysis

```text
1. 단일 경량 모델(KLUE / KoELECTRA)은
   대형 LLM 대비 파라미터 수가 적음에도 불구하고
   안정적인 기준 성능(~0.7)을 달성함.

2. 4-Fold Soft Voting Ensemble 적용 시,
   Fold 간 분산이 감소하며
   Macro F1 기준 성능이 약 0.84까지 향상됨.

3. Hard Negative 중심 Downsampling,
   Sliding Window 기반 Long Text 학습,
   Label Smoothing을 통한 편향 완화가
   성능 향상에 핵심적으로 기여함.
```

## 7️⃣ Reproducibility & Environment

본 프로젝트는 **실험 재현성을 최우선으로 고려하여**  
동일한 데이터 분할, 동일한 난수 시드, 명시적인 실행 환경을 기반으로 수행되었다.

---

### Environment

#### Hardware / OS
- **Host OS:** Windows 11  
- **Execution Environment:** WSL2 (Ubuntu 24.04)
- **GPU:** AMD Radeon RX 7800 XT (VRAM 16GB)

#### Software
- **Python:** 3.10
- **PyTorch:** 2.6.0
- **HuggingFace Transformers:** 4.46.3

---

### Reproducibility Settings

```text
- 동일한 4-Fold 데이터 분할 사용
- Random Seed 고정
- 학습 / 평가 파이프라인 전 과정에서 동일 설정 유지
```

📌 모든 실험은 동일한 Fold 분할 및 Seed 고정 환경에서 수행되었습니다.

## 8️⃣ Repository Structure

```text
make-model/
├── data/
│   ├── raw/        # 원본 데이터
│   ├── interim/    # 정제 데이터
│   └── fold/       # 4-Fold 분할 데이터
├── notebooks/
│   ├── EDA.ipynb
│   ├── Data Preprocessing.ipynb
│   ├── Klue_roberta-small.ipynb
│   ├── koelectra-small.ipynb
│   └── Ensemble.ipynb
├── model/
│   └── result_models/
├── eda.py
├── requirements.txt
└── README.md
```

⚠️ 모든 Notebook은 프로젝트 루트 기준 경로로 작성되었습니다.

## 9️⃣ License

MIT License
This project is licensed under the MIT License.
