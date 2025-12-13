import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import time
import easyocr
from PIL import Image

# ==========================================
# 1. 기본 설정 및 상수
# ==========================================
st.set_page_config(
    page_title="On-premise AI Detector",
    page_icon="🛡️",
    layout="wide"
)

# 모델 경로 (사용자 환경에 맞게 수정 필요)
MODEL_PATH = "experiments/koelectra_small_sliding_single_test"

# ==========================================
# 2. 리소스 로드 함수 (Caching 적용)
# ==========================================

# 2-1. KoELECTRA 모델 로드
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
        print(f"Error loading model: {e}")
        return None, None, None

# 2-2. EasyOCR 모델 로드 (Vision 기능용)
@st.cache_resource
def load_ocr_reader():
    # GPU 사용 가능 여부 확인
    use_gpu = torch.cuda.is_available()
    print(f"OCR Loading... GPU: {use_gpu}")
    # 한국어(ko), 영어(en) 인식 모델 로드
    return easyocr.Reader(['ko', 'en'], gpu=use_gpu)

# 전역 리소스 초기화
tokenizer, model, device = load_model_and_tokenizer(MODEL_PATH)
ocr_reader = load_ocr_reader()

# ==========================================
# 3. 실제 추론 함수 (Core Logic)
# ==========================================
def predict_text(text, tokenizer, model, device):
    if not text or model is None:
        return 0.0, 0.0, 0.0

    start_time = time.time()
    
    # 토큰화 및 텐서 변환
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(device)

    # 추론
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # 확률 계산 (Softmax)
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    
    end_time = time.time()
    latency = round((end_time - start_time) * 1000, 2) # ms 단위

    human_prob = probs[0]
    ai_prob = probs[1]

    return human_prob, ai_prob, latency

# ==========================================
# 4. 페이지별 UI 함수
# ==========================================

def page_home():
    st.title("🛡️ On-premise AI Text Detector")
    st.markdown("### 고성능 경량화 모델 기반의 텍스트 분석 솔루션")
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 🔒 Privacy First\n내부 서버 구동으로 데이터 유출 걱정 없음")
    with col2:
        st.success("### 💸 Cost Efficiency\nAPI 비용 없는 저렴한 유지비용")
    with col3:
        st.warning("### 🚀 High Performance\nKoELECTRA 모델 기반 정밀 분석")

    st.divider()
    
    # 시스템 상태 표시
    st.subheader("System Status")
    c1, c2 = st.columns(2)
    with c1:
        if device and device.type == 'cuda':
            st.success(f"Running on GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("Running on CPU")
    with c2:
        if ocr_reader:
            st.success("OCR Engine: Ready")
        else:
            st.error("OCR Engine: Failed")

def page_lms():
    st.header("🎓 LMS 탑재형 과제 검수기")
    st.markdown("**에세이를 분석하여 AI 작성 여부를 스크리닝합니다.**")
    
    if model is None:
        st.error("모델이 로드되지 않았습니다. 경로를 확인해주세요.")
        return

    with st.form("lms_form"):
        text = st.text_area("과제 내용 입력", height=300, placeholder="학생이 제출한 에세이를 입력하세요...")
        submitted = st.form_submit_button("검사 수행")
        
    if submitted and text:
        with st.spinner("Deep Learning Model Analyzing..."):
            human_prob, ai_prob, latency = predict_text(text, tokenizer, model, device)
            word_count = len(text.split())
            
        # 결과 판정 로직
        if ai_prob >= 0.85:
            status = "red_flag"
            label_msg = "High Risk (AI 의심)"
        elif ai_prob >= 0.50:
            status = "warning"
            label_msg = "Medium Risk (검토 필요)"
        else:
            status = "clear"
            label_msg = "Low Risk (사람 작성)"
            
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("단어 수", f"{word_count} words")
            st.metric("AI 확률", f"{ai_prob*100:.1f}%")
            st.caption(f"Latency: {latency}ms")
            
        with c2:
            if status == "red_flag":
                st.error(f"🚩 **RED FLAG 감지됨**\n\nAI 작성 패턴이 강하게 의심됩니다 ({label_msg}).\n**정밀 검토가 필요합니다.**")
            elif status == "warning":
                st.warning(f"⚠️ **주의 요망**\n\n일부 문장이 부자연스럽거나 AI 패턴이 섞여있습니다 ({label_msg}).")
            else:
                st.success(f"✅ **통과 (Clear)**\n\n사람이 작성한 것으로 추정됩니다 ({label_msg}).")
            
            st.write("#### 상세 분석")
            st.progress(int(ai_prob * 100), text=f"AI Score: {ai_prob*100:.1f}%")
            st.progress(int(human_prob * 100), text=f"Human Score: {human_prob*100:.1f}%")

def page_spam():
    st.header("🚨 실시간 스팸/피싱 탐지기 (Vision)")
    st.markdown("**문자 내용을 입력하거나, 스크린샷/카메라로 찍으면 즉시 분석합니다.**")
    
    if model is None:
        st.error("모델 오류: 로드 실패")
        return

    # 탭으로 입력 방식 분리
    tab1, tab2 = st.tabs(["📝 텍스트 직접 입력", "📸 스크린샷/카메라 분석"])
    
    target_text = ""
    is_image_processed = False

    # [Tab 1] 텍스트 입력
    with tab1:
        user_input = st.text_area("메시지 내용", height=150, placeholder="[Web발신] 당첨을 축하합니다...", key="text_input_area")
        if st.button("텍스트 분석 실행", key="btn_text"):
            target_text = user_input

    # [Tab 2] OCR (이미지 분석) - PC 붙여넣기 가이드 추가됨
    with tab2:
        # PC 사용자용 가이드
        st.info("""
        💡 **PC 사용자 캡처 팁:**
        1. **`Win` + `Shift` + `S`** 키로 캡처
        2. 아래 **'Browse files'** 영역 클릭
        3. **`Ctrl` + `V`** 로 붙여넣기
        """)
        
        # 파일 업로더와 카메라 입력을 동시에 지원
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            # 라벨에 (Ctrl+V) 명시
            img_file = st.file_uploader("이미지 붙여넣기(Ctrl+V) 또는 업로드", type=['png', 'jpg', 'jpeg'])
        with col_img2:
            camera_input = st.camera_input("카메라로 찍기 (모바일)")
        
        # 우선순위: 카메라 -> 파일 업로드
        target_image = camera_input if camera_input else img_file
        
        if target_image:
            # 이미지 보여주기
            st.image(target_image, caption="분석 대상 이미지", width=400)
            
            if st.button("이미지 분석 시작", type="primary", key="btn_ocr"):
                with st.spinner("📷 이미지에서 글자를 읽어내는 중... (OCR)"):
                    try:
                        image = Image.open(target_image)
                        image_np = np.array(image)
                        
                        # EasyOCR로 텍스트 추출 (detail=0은 텍스트 리스트만 반환)
                        result = ocr_reader.readtext(image_np, detail=0)
                        target_text = " ".join(result)
                        is_image_processed = True
                        
                        if not target_text.strip():
                            st.warning("이미지에서 텍스트를 인식하지 못했습니다.")
                    except Exception as e:
                        st.error(f"OCR 처리 중 오류 발생: {e}")

    # [공통] 분석 실행 및 결과 출력
    if target_text:
        st.divider()
        if is_image_processed:
            with st.expander("🔍 이미지에서 추출된 텍스트 보기", expanded=True):
                st.text(target_text)
        
        with st.spinner("🤖 AI 모델이 스팸 여부를 판단 중입니다..."):
            human_prob, ai_prob, latency = predict_text(target_text, tokenizer, model, device)

        # 결과 카드 디자인
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("AI 스팸 확률", f"{ai_prob*100:.1f}%")
            
        with col2:
            if ai_prob >= 0.85:
                st.error(f"🚫 **위험 (DANGER)**\n\n피싱/스팸일 확률이 매우 높습니다!")
            elif ai_prob >= 0.50:
                st.warning(f"⚠️ **주의 (WARNING)**\n\n의심스러운 문구가 포함되어 있습니다.")
            else:
                st.success(f"✅ **안전 (SAFE)**\n\n정상적인 메시지로 보입니다.")
        
        st.caption(f"📊 분석 모델: KoELECTRA Custom | ⚡ Latency: **{latency}ms**")
        
        with st.expander("개발자용 디버그 정보"):
            st.json({
                "source_length": len(target_text),
                "human_prob": float(human_prob),
                "ai_prob": float(ai_prob),
                "ocr_used": is_image_processed
            })

# ==========================================
# 5. 메인 컨트롤러
# ==========================================
def main():
    with st.sidebar:
        st.title("🔧 솔루션 모드 선택")
        choice = st.radio("Mode", ["프로젝트 소개", "LMS 과제 검수", "실시간 스팸 탐지"])
        
        st.divider()
        st.markdown("### System Info")
        if device:
            st.caption(f"Device: {device}")
            st.caption(f"Model: {MODEL_PATH.split('/')[-1]}")
        else:
            st.error("Model Load Failed")
        
        st.markdown("---")
        st.caption("Developed with Streamlit & EasyOCR")

    if choice == "프로젝트 소개":
        page_home()
    elif choice == "LMS 과제 검수":
        page_lms()
    elif choice == "실시간 스팸 탐지":
        page_spam()

if __name__ == "__main__":
    main()