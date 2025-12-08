import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# ---------------------------------------------------------
# 1. 설정
# ---------------------------------------------------------
# checkpoint 폴더 지정
MODEL_PATH = "make-model/experiments/koelectra_small_sliding_single_test" 
LABELS = {0: "Human (사람)", 1: "AI Generated (인공지능)"}

# ---------------------------------------------------------
# 2. 모델 로드 함수 (cashing 적용)
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_tokenizer(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"모델을 불러오는 중 오류가 발생했습니다: {e}")
        return None, None, None

# ---------------------------------------------------------
# 3. 메인 UI 및 추론 로직
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Text Detector", page_icon="#")
    
    st.title("AI 문장 판별기")
    st.markdown("입력한 텍스트가 사람이 쓴 것인지 AI가 쓴 것인지 분석합니다.")

    # 사이드바: 모델 로드 상태 표시
    with st.sidebar:
        st.header("System Status")
        if torch.cuda.is_available():
            st.success(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("Running on CPU")

    # 모델 로드
    tokenizer, model, device = load_model_and_tokenizer(MODEL_PATH)

    if model is None:
        st.warning(f"경로를 확인해주세요: {MODEL_PATH}")
        st.stop()

    # 텍스트 입력
    text_input = st.text_area("분석할 텍스트를 입력하세요:", height=200, placeholder="여기에 글을 붙여넣으세요...")

    if st.button("분석 시작", type="primary"):
        if not text_input.strip():
            st.warning("텍스트를 입력해주세요")
            return

        with st.spinner("분석 중입니다..."):
            inputs = tokenizer(
                text_input, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512, 
                padding="max_length"
            ).to(device)

            # 2. 추론
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            # 3. 확률 계산 (Softmax)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx] * 100

            # 4. 결과 출력
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if pred_idx == 1: 
                    st.error(f"- AI 작성 의심")
                else: 
                    st.success(f"- 사람 작성 추정")
            
            with col2:
                st.metric(label="확신도 (Confidence)", value=f"{confidence:.2f}%")
            
            # 5. 상세 확률 바 차트
            st.write("### 상세 분석 결과")
            st.progress(int(probs[1] * 100), text=f"AI 확률: {probs[1]*100:.2f}%")
            st.progress(int(probs[0] * 100), text=f"Human 확률: {probs[0]*100:.2f}%")


            with st.expander("Raw Output 보기"):
                st.json({"Human_Prob": float(probs[0]), "AI_Prob": float(probs[1])})

if __name__ == "__main__":
    main()