import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import time
import os

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="AI-ColoScan Pro", layout="wide")

# Инициализация хранилища (Session State)
if 'top_crops' not in st.session_state:
    st.session_state.top_crops = []
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = 0

@st.cache_resource
def load_model():
    # Загружаем модель один раз и держим в кэше
    return YOLO('kvasir+polypDB.pt')

model = load_model()

# CSS для дизайна и плавности
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .img-label { text-align: center; font-size: 16px; font-weight: 700; color: #3b82f6; margin-bottom: 5px; }
    .status-box {
        padding: 20px; border-radius: 10px; text-align: center;
        font-size: 28px; font-weight: 900; margin: 10px 0; transition: 0.3s;
    }
    .found { background-color: #7f1d1d; color: #f87171; border: 2px solid #ef4444; box-shadow: 0 0 10px #ef4444; }
    .not-found { background-color: #064e3b; color: #34d399; border: 2px solid #10b981; }
    </style>
    """, unsafe_allow_html=True)

st.title(" AI-ColoScan: Clinical Diagnostic System")
st.write("Optimized version (Safe Mode Enabled)")
st.divider()

uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'], label_visibility="collapsed")

if uploaded_file is not None:
    # Сохраняем во временный файл безопасно
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown('<div class="img-label">ORIGINAL FEED</div>', unsafe_allow_html=True)
        raw_placeholder = st.empty()
    with col_v2:
        st.markdown('<div class="img-label">AI DIAGNOSIS</div>', unsafe_allow_html=True)
        proc_placeholder = st.empty()

    status_placeholder = st.empty()
    stop_btn = st.button("STOP ANALYSIS", use_container_width=True)

    frame_count = 0
    # ПАРАМЕТР ОПТИМИЗАЦИИ: обрабатываем каждый N-й кадр
    SKIP_FRAMES = 3 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_btn:
            break
        
        frame_count += 1
        
        # Пропускаем кадры для снижения нагрузки на CPU (защита от блокировки 403)
        if frame_count % SKIP_FRAMES != 0:
            continue

        current_timestamp = time.strftime('%M:%S', time.gmtime(frame_count / fps))
        
        # Уменьшаем размер кадра перед подачей в модель (еще больше оптимизации)
        small_frame = cv2.resize(frame, (640, 480))
        results = model.predict(small_frame, conf=0.5, verbose=False)
        
        # Вывод видео (сжатое для экономии трафика Streamlit)
        f_raw_disp = cv2.cvtColor(cv2.resize(frame, (400, 260)), cv2.COLOR_BGR2RGB)
        raw_placeholder.image(f_raw_disp)
        
        f_proc = results[0].plot()
        f_proc_disp = cv2.cvtColor(cv2.resize(f_proc, (400, 260)), cv2.COLOR_BGR2RGB)
        proc_placeholder.image(f_proc_disp)
        
        is_detected = len(results[0].boxes) > 0
        
        if is_detected:
            st.session_state.last_detection_time = time.time()
            
            box = results[0].boxes[0]
            conf = box.conf[0].item()
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            # Делаем кроп аккуратно
            crop = small_frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            if crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                # Обновляем ТОП-5
                if len(st.session_state.top_crops) < 5 or conf > min(st.session_state.top_crops, key=lambda x: x[0])[0]:
                    # Проверка на дубликаты по времени (не чаще раза в 2 секунды)
                    if not st.session_state.top_crops or current_timestamp != st.session_state.top_crops[-1][2]:
                        st.session_state.top_crops.append((conf, crop_rgb, current_timestamp))
                        st.session_state.top_crops = sorted(st.session_state.top_crops, key=lambda x: x[0], reverse=True)[:5]

        # Умный статус (анти-мерцание)
        if is_detected or (time.time() - st.session_state.last_detection_time < 2.0):
            status_placeholder.markdown('<div class="status-box found"> POLYP DETECTED</div>', unsafe_allow_html=True)
        else:
            status_placeholder.markdown('<div class="status-box not-found"> NO POLYPS</div>', unsafe_allow_html=True)

    cap.release()
    # Удаляем временный файл после работы
    if os.path.exists(video_path):
        os.remove(video_path)

    # --- КАРУСЕЛЬ ТАБЫ ---
    st.divider()
    st.subheader("Diagnostic Highlights (Top 5 Results)")
    
    if st.session_state.top_crops:
        tab_titles = [f"Result {i+1} at {item[2]}" for i, item in enumerate(st.session_state.top_crops)]
        tabs = st.tabs(tab_titles)
        
        for i, item in enumerate(st.session_state.top_crops):
            with tabs[i]:
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(item[1], use_container_width=True)
                with c2:
                    st.metric("AI Confidence", f"{item[0]*100:.1f}%")
                    st.warning(f"Detected at video timestamp: {item[2]}")
                    st.write("Review the original footage at this timestamp for clinical confirmation.")
    else:
        st.write("No pathologies found during this session.")
else:
    st.info("System Ready. Please upload endoscopy footage to start.")
