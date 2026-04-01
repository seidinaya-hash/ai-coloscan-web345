import streamlit as st
import os

# Принудительное отключение графического интерфейса OpenCV
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import time

st.set_page_config(page_title="AI-ColoScan Pro", layout="wide")

if 'top_crops' not in st.session_state:
    st.session_state.top_crops = [] 
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = 0

@st.cache_resource
def load_model():
    return YOLO('kvasir+polypDB.pt')

model = load_model()

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .img-label { text-align: center; font-size: 16px; font-weight: 700; color: #3b82f6; margin-bottom: 5px; }
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
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_btn:
            break
        
        frame_count += 1
        if frame_count % 3 != 0:
            continue
            
        current_timestamp = time.strftime('%M:%S', time.gmtime(frame_count / fps))
        results = model.predict(frame, conf=0.5, verbose=False)
        
        f_raw = cv2.cvtColor(cv2.resize(frame, (480, 320)), cv2.COLOR_BGR2RGB)
        raw_placeholder.image(f_raw)
        
        f_proc = results[0].plot()
        f_proc = cv2.cvtColor(cv2.resize(f_proc, (480, 320)), cv2.COLOR_BGR2RGB)
        proc_placeholder.image(f_proc)
        
        is_detected = len(results[0].boxes) > 0
        if is_detected:
            st.session_state.last_detection_time = time.time()
            box = results[0].boxes[0]
            conf = box.conf[0].item()
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            crop = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            if crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                if len(st.session_state.top_crops) < 5 or conf > min(st.session_state.top_crops, key=lambda x: x[0])[0]:
                    st.session_state.top_crops.append((conf, crop_rgb, current_timestamp))
                    st.session_state.top_crops = sorted(st.session_state.top_crops, key=lambda x: x[0], reverse=True)[:5]

        if is_detected or (time.time() - st.session_state.last_detection_time < 1.5):
            status_placeholder.markdown('<div class="status-box found">POLYP DETECTED</div>', unsafe_allow_html=True)
        else:
            status_placeholder.markdown('<div class="status-box not-found">NO POLYPS</div>', unsafe_allow_html=True)

    cap.release()

    st.divider()
    st.subheader("Diagnostic Highlights (Top 5)")
    if st.session_state.top_crops:
        tab_titles = [f"Detection {i+1} ({item[2]})" for i, item in enumerate(st.session_state.top_crops)]
        tabs = st.tabs(tab_titles)
        for i, item in enumerate(st.session_state.top_crops):
            with tabs[i]:
                c_col1, c_col2 = st.columns([1, 2])
                with c_col1:
                    st.image(item[1], use_container_width=True)
                with c_col2:
                    st.metric("Certainty", f"{item[0]*100:.1f}%")
                    st.write(f"Timestamp: {item[2]}")
    else:
        st.write("No suspicious areas found.")
else:
    st.info("Upload endoscopic video to begin.")
