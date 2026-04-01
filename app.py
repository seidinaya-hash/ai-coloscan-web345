import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="AI-ColoScan PRO", layout="wide")

# Инициализация хранилища для топ-5 находок
if 'top_finds' not in st.session_state:
    st.session_state.top_finds = [] # Список кортежей (conf, image)

# Загрузка модели
@st.cache_resource
def load_model():
    return YOLO('kvasir+polypDB.pt')

model = load_model()

# CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e6edf3; }
    .img-label { 
        text-align: center; 
        font-size: 16px; 
        font-weight: 700; 
        color: #3b82f6; 
        text-transform: uppercase; 
        margin-bottom: 5px; 
    }
    .custom-card { 
        background-color: #1c2533; 
        border: 1px solid #3b82f6; 
        border-radius: 10px; 
        padding: 10px; 
        text-align: center; 
    }
    .custom-label { color: #94a3b8; font-size: 12px; text-transform: uppercase; }
    .custom-value { color: #ffffff; font-size: 24px; font-weight: 900; }
    .top-find-img { border-radius: 5px; border: 1px solid #3b82f6; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 AI-ColoScan: Video Diagnostic System")
st.write("Upload endoscopy video for real-time polyp detection and sizing.")
st.divider()

# --- ЗАГРУЗКА ТОЛЬКО ВИДЕО ---
uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'], label_visibility="collapsed")

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # Центрированная сетка для видео (делаем их меньше за счет пустых колонок по бокам)
    _, col_v1, col_v2, _ = st.columns([1, 4, 4, 1])
    
    with col_v1:
        st.markdown('<div class="img-label">INPUT VIDEO FEED</div>', unsafe_allow_html=True)
        raw_video_placeholder = st.empty()
    with col_v2:
        st.markdown('<div class="img-label">PROCESSED AI DIAGNOSIS</div>', unsafe_allow_html=True)
        processed_video_placeholder = st.empty()

    # Панель управления и метрики
    st.divider()
    m1, m2, m3 = st.columns(3)
    
    stop_button = st.button("STOP ANALYSIS", use_container_width=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_button:
            break
        
        # Инференс модели
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # Оригинальное видео (уменьшаем для скорости и экономии места)
        frame_rgb = cv2.cvtColor(cv2.resize(frame, (480, 360)), cv2.COLOR_BGR2RGB)
        raw_video_placeholder.image(frame_rgb)
        
        # Обработанное видео
        annotated_frame = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(cv2.resize(annotated_frame, (480, 360)), cv2.COLOR_BGR2RGB)
        processed_video_placeholder.image(annotated_frame_rgb)
        
        # Если найден полип
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            conf = box.conf[0].item()
            
            # Примерный расчет размера (симуляция на основе площади бокса)
            # В реальности нужно калибровать камеру
            w = box.xywh[0][2].item()
            h = box.xywh[0][3].item()
            estimated_size = (w + h) / 20 # Условный коэффициент
            
            # Обновление карточек
            m1.markdown(f'<div class="custom-card"><span class="custom-label">Status</span><span class="custom-value">POLYP FOUND</span></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="custom-card"><span class="custom-label">Estimated Size</span><span class="custom-value">{estimated_size:.1f} mm</span></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="custom-card"><span class="custom-label">Certainty</span><span class="custom-value">{conf*100:.1f}%</span></div>', unsafe_allow_html=True)
            
            # Логика Топ-5 находок
            if len(st.session_state.top_finds) < 5 or conf > min(st.session_state.top_finds, key=lambda x: x[0])[0]:
                # Добавляем кадр и сортируем
                st.session_state.top_finds.append((conf, annotated_frame_rgb))
                st.session_state.top_finds = sorted(st.session_state.top_finds, key=lambda x: x[0], reverse=True)[:5]
        else:
            m1.markdown(f'<div class="custom-card"><span class="custom-label">Status</span><span class="custom-value">CLEAR</span></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="custom-card"><span class="custom-label">Estimated Size</span><span class="custom-value">0 mm</span></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="custom-card"><span class="custom-label">Certainty</span><span class="custom-value">0%</span></div>', unsafe_allow_html=True)

    cap.release()

    # --- КАРУСЕЛЬ ТОП-5 НАХОДОК ---
    st.divider()
    st.subheader("Top 5 Detections (Largest/Most Certain)")
    if st.session_state.top_finds:
        cols = st.columns(5)
        for i, (score, img_data) in enumerate(st.session_state.top_finds):
            with cols[i]:
                st.image(img_data, caption=f"Confidence: {score*100:.1f}%", use_container_width=True)
    else:
        st.write("No detections yet.")

else:
    st.info("Please upload an endoscopy video file (MP4, MOV, AVI) to begin.")
