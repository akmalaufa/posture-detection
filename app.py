import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
from datetime import timedelta

# Load model, scaler, dan label encoder
model = joblib.load("best-model/mlp_posture_model.pkl")
scaler = joblib.load("best-model/scaler.pkl")
label_encoder = joblib.load("best-model/label_encoder.pkl")

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Fungsi ekstraksi 18 fitur (9 titik x, y)
def extract_keypoints(results):
    indices = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.RIGHT_EYE
    ]
    if not results.pose_landmarks:
        return np.zeros(18)
    return np.array([coord
                     for idx in indices
                     for coord in (results.pose_landmarks.landmark[idx].x,
                                   results.pose_landmarks.landmark[idx].y)])

# Threshold durasi peringatan
REMINDER_THRESHOLD = 30 * 60  # 30 menit

# Inisialisasi session state
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "last_posture" not in st.session_state:
    st.session_state.last_posture = None
    st.session_state.start_time = time.time()
    st.session_state.duration = 0.0
    st.session_state.warning_shown = False

# UI Layout
st.set_page_config(layout="wide", page_title="Prediksi Postur Tubuh Real-Time")
st.title("📸 Prediksi Postur Tubuh Real-Time")

# Tombol Start / Stop Kamera
col1, col2 = st.columns([1, 4])
with col1:
    if not st.session_state.camera_on:
        if st.button("▶️ Start Kamera"):
            st.session_state.camera_on = True
            st.rerun()
    else:
        if st.button("⏹️ Stop Kamera"):
            st.session_state.camera_on = False
            st.rerun()

# Sidebar tampilan info
st.sidebar.header("📊 Informasi Postur")
posture_text = st.sidebar.empty()
duration_text = st.sidebar.empty()
notification_area = st.sidebar.empty()

# Placeholder video
webcam_placeholder = st.empty()

# Main logic webcam
if st.session_state.camera_on:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Webcam tidak bisa diakses.")
            break

        # Proses frame
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        keypoints = extract_keypoints(results)
        scaled = scaler.transform([keypoints])
        pred = model.predict(scaled)
        posture = label_encoder.inverse_transform(pred)[0]

        # Gambar landmark pada frame
        annotated = image.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Hitung durasi postur
        now = time.time()
        if posture == st.session_state.last_posture:
            st.session_state.duration = now - st.session_state.start_time
        else:
            st.session_state.last_posture = posture
            st.session_state.start_time = now
            st.session_state.duration = 0.0
            st.session_state.warning_shown = False

        # Update UI
        webcam_placeholder.image(annotated, channels="RGB")
        posture_text.markdown(f"**Postur Saat Ini:** `{posture}`")
        durasi_str = str(timedelta(seconds=int(st.session_state.duration)))
        duration_text.markdown(f"**Durasi Postur:** `{durasi_str}`")

        # NOTIFIKASI: hanya jika postur BUKAN "looks good"
        if (
            st.session_state.duration >= REMINDER_THRESHOLD and
            not st.session_state.warning_shown and
            posture.lower() != "looks good"
        ):
            notification_area.warning(
                f"⚠️ Anda telah berada dalam postur **'{posture}'** selama lebih dari 30 menit. Silakan ubah posisi.")
            st.session_state.warning_shown = True

        time.sleep(0.03)

    # Setelah stop kamera
    cap.release()
    cv2.destroyAllWindows()
    webcam_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Kamera tidak aktif", channels="RGB")
    posture_text.markdown("**Postur Saat Ini:** `-`")
    duration_text.markdown("**Durasi Postur:** `00:00:00`")
    notification_area.empty()
