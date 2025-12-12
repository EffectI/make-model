# 🤖 AI-Text-Classifier  
### Lightweight Human–AI Text Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔑 Executive Summary (한 문단 요약)

본 프로젝트는 **AI 생성 텍스트와 인간 작성 텍스트를 구분**하기 위한  
**경량화(Lightweight) 모델 기반 분류 파이프라인**을 구축한다.

EDA를 통해 **Perplexity·Entropy·문체 변동성**이 핵심 신호임을 검증하였으며,  
구분력이 낮은 Feature는 명시적으로 배제하였다.  
최종적으로 **4-Fold 교차검증 + Soft Voting 앙상블**을 적용하여  
재현 가능하고 일반화 성능이 높은 구조를 목표로 한다.

---

## 📌 Project Motivation & Scope

최근 LLM의 발전으로 AI 생성 텍스트는 문법·어휘 측면에서 인간과 거의 구분되지 않는다.  
그러나 **“예측 가능성, 변동성, 반복성”** 관점에서 보면 여전히 구조적 차이가 존재한다.

본 프로젝트는 다음 질문에서 출발한다.

> **“대형 모델 없이도, 통계적·문체적 신호만으로 AI 텍스트를 식별할 수 있는가?”**

이를 위해:
- 대형 LLM API ❌  
- **경량 Transformer + EDA 기반 Feature 선택 ⭕**

라는 전략을 채택하였다.

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

- Total documents: **97,172**
- Columns:
  - `title` (string)
  - `full_text` (string)
  - `generated` (0 = Human, 1 = AI)

### Class Distribution

- Human (0): 89,177  
- AI (1): 7,995  
- Ratio ≈ **11 : 1 (Severely Imbalanced)**

📌 **Implication:**  
Accuracy 단독 평가는 부적절하며,  
**Macro F1-Score** 기반 평가 및  
downsampling / class weighting 전략이 필요하다.

---

## 2️⃣ Exploratory Data Analysis (EDA) – Decision-Driven Summary

EDA의 목적은 **Feature 후보의 생존 여부를 판단**하는 것이다.  
분석 결과 Feature들은 다음 세 그룹으로 명확히 구분되었다.

---

### ✅ Strong Signal

#### ▸ Perplexity & Entropy (Language-Model-based)

- Human 텍스트:
  - Perplexity 분포 폭이 넓음
  - 예측 불확실성(Entropy) 높음
- AI 텍스트:
  - Perplexity 낮고 분포가 매우 안정적

📌 **결론:**  
→ 가장 강력한 분리 신호  
→ 본 프로젝트의 핵심 Feature

---

### ⚠️ Medium Signal

#### ▸ Text Length / Lexical Diversity (TTR) / Stylistic Variance

- Human:
  - 문서 길이, 문장 길이, 어휘 분포의 변동성 큼
- AI:
  - 특정 길이 구간에 집중
  - 반복적 연결어·문장 패턴

📌 **결론:**  
→ 단독 Feature로는 약함  
→ 다른 Feature와 결합 시 보조적 성능 향상

---

### ❌ Weak Signal (Discarded)

#### ▸ Special Character Patterns

- 한자, HTML tag, 반복 괄호·마침표·쉼표 등
- 모든 패턴에서:
  - 절대값 ≈ 0
  - Cohen’s d < 0.1

📌 **결론:**  
→ 통계적으로 유의미하지 않음  
→ Feature로 사용하지 않고 **전처리 단계에서만 활용**

📎 모든 EDA 시각화·통계는 `notebooks/EDA.ipynb`에 보존됨.

---

## 3️⃣ Feature Evaluation & Selection (Keep vs Discard)

| Feature Category | Decision | Rationale |
|------------------|----------|-----------|
| Perplexity / Entropy | ✔ Keep | 가장 강력한 분리 신호 |
| Text Length | ✔ Keep | Weak classifier 가능 |
| Lexical Diversity (TTR) | ✔ Keep | 문체 변동성 반영 |
| Stylistic Metrics | ✔ Keep | 구조적 반복성 탐지 |
| Special Characters | ✘ Discard | 효과크기 미미 |

📌 모든 Feature 선택은 **실험 기반으로만 결정**되었으며,  
직관이나 추측에 의존한 Feature는 존재하지 않는다.

---

## 4️⃣ Modeling Strategy

### Backbone Models (Lightweight)

- `klue/roberta-small`
- `monologg/koelectra-small`

**선정 이유**
- 한국어 사전학습 모델
- 파라미터 수 대비 표현력 우수
- 빠른 학습 및 추론 가능

### Validation Strategy

- **4-Fold Cross Validation**
- Fold별 모델을 **Soft Voting Ensemble**로 통합

📌 **Why Ensemble?**
- 단일 모델 편향 감소
- Fold 간 분산 완화
- 일반화 성능 향상

---

---

## 5️⃣ Training & Inference Pipeline

### Step 1. Preprocessing & Fold Generation

### Step 2. Model Training (per Fold)

각 Fold별로 아래 노트북을 사용하여 모델을 학습합니다.

- Klue_roberta-small.ipynb
- koelectra-small.ipynb

학습이 완료된 모델 가중치는 다음 경로에 저장됩니다.

model/result_models/

### Step 3. Ensemble & Prediction

- 사용 노트북: Ensemble.ipynb
- 방식: Soft Voting Ensemble
- 최종 결과물: submission.csv

## 6️⃣ Experiments & Evaluation

### Evaluation Results (Work in Progress)

| Model | Backbone | Metric | Status |
|------|----------|--------|--------|
| KLUE RoBERTa | roberta-small | Macro F1 | WIP |
| KoELECTRA | electra-small | Macro F1 | WIP |
| Ensemble | Soft Voting | Macro F1 | **Best** |

### Evaluation Metrics

- **Macro F1-Score** (Primary)
- **Accuracy** (Secondary)


## 7️⃣ Reproducibility & Environment

### Environment

- Python **3.10+**
- PyTorch **2.0+**
- HuggingFace Transformers

### Installation

📌 모든 실험은 동일한 Fold 분할 및 Seed 고정 환경에서 수행되었습니다.

## 8️⃣ Repository Structure
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


⚠️ 모든 Notebook은 프로젝트 루트 기준 경로로 작성되었습니다.

## 9️⃣ License

MIT License
This project is licensed under the MIT License.
