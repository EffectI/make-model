![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow) ![License](https://img.shields.io/badge/License-MIT-green)
````markdown
# 🤖 AI-Text-Classifier (Lightweight Model Project)


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