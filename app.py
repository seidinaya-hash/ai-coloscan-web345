import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tempfile
from ultralytics import YOLO
import time

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="AI-ColoScan Pro", layout="wide")

if 'top_crops' not in st.session_state:
    st.session_state.top_crops = [] 
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = 0

@st.cache_resource
def load_model():
    # Загружаем модель
    return YOLO('kvasir+polypDB.pt')

model = load_model()

# CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .status-box {
        padding: 20px; border-radius: 10px; text-align: center;
        font-size: 30px; font-weight: 900; margin: 10px 0;
    }
    .found { background-color: #7f1d1d; color: #f87171; border: 2px solid #ef4444; }
    .not-found { background-color: #064e3b; color: #34d399; border: 2px solid #10b981; }
    </style>
    """, unsafe_allow_html=True)

st.title("AI-ColoScan: Clinical Diagnostic System")
st.divider()

uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'], label_visibility="collapsed")

if uploaded_file is not None:
    # Используем встроенный проигрыватель Streamlit для отображения видео
    # А обработку будем делать покадрово через библиотеку decord или av (более легкие)
    # Но для стабильности используем дефолтный подход с image_set
    
    import cv2 # Мы пробуем импортировать его внутри, чтобы если он упадет, упала только часть
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.write("ORIGINAL FEED")
        raw_placeholder = st.empty()
    with col_v2:
        st.write("AI DIAGNOSIS")
        proc_placeholder = st.empty()

    status_placeholder = st.empty()
    stop_btn = st.button("STOP ANALYSIS", use_container_width=True)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_btn:
            break
        
        frame_count += 1
        if frame_count % 3 != 0: # Пропускаем кадры для скорости
            continue
            
        current_timestamp = time.strftime('%M:%S', time.gmtime(frame_count / fps))
        
        # Конвертируем кадр в PIL Image (это не требует libGL)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Инференс
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # Рисуем рамки сами через PIL (без cv2.rectangle)
        draw_img = img_pil.copy()
        draw = ImageDraw.Draw(draw_img)
        
        is_detected = len(results[0].boxes) > 0
        
        if is_detected:
            st.session_state.last_detection_time = time.time()
            for box in results[0].boxes:
                b = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                # Рисуем рамку
                draw.rectangle([b[0], b[1], b[2], b[3]], outline="red", width=5)
                
                # Сохраняем кроп
                if len(st.session_state.top_crops) < 5 or conf > min(st.session_state.top_crops, key=lambda x: x[0])[0]:
                    crop = img_pil.crop((b[0], b[1], b[2], b[3]))
                    st.session_state.top_crops.append((conf, crop, current_timestamp))
                    st.session_state.top_crops = sorted(st.session_state.top_crops, key=lambda x: x[0], reverse=True)[:5]

        # Вывод изображений
        raw_placeholder.image(img_pil.resize((480, 320)))
        proc_placeholder.image(draw_img.resize((480, 320)))

        # Статус
        if is_detected or (time.time() - st.session_state.last_detection_time < 1.5):
            status_placeholder.markdown('<div class="status-box found">POLYP DETECTED</div>', unsafe_allow_html=True)
        else:
            status_placeholder.markdown('<div class="status-box not-found">NO POLYPS</div>', unsafe_allow_html=True)

    cap.release()

    # Карусель
    st.divider()
    st.subheader("Diagnostic Highlights")
    if st.session_state.top_crops:
        tabs = st.tabs([f"Detection {i+1} ({item[2]})" for i, item in enumerate(st.session_state.top_crops)])
        for i, item in enumerate(st.session_state.top_crops):
            with tabs[i]:
                c1, c2 = st.columns([1, 2])
                c1.image(item[1])
                c2.metric("Certainty", f"{item[0]*100:.1f}%")
                c2.write(f"Timestamp: {item[2]}")
else:
    st.info("Upload video to begin.")
