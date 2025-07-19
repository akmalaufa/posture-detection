import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
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

REMINDER_THRESHOLD = 30 * 60  # 30 menit

# Session state untuk durasi dan notifikasi
if "last_posture" not in st.session_state:
    st.session_state.last_posture = None
    st.session_state.start_time = time.time()
    st.session_state.duration = 0.0
    st.session_state.warning_shown = False

# Sidebar tampilan info
st.set_page_config(layout="wide", page_title="Prediksi Postur Tubuh Real-Time (WebRTC)")
st.title("📸 Prediksi Postur Tubuh Real-Time (WebRTC)")
st.sidebar.header("📊 Informasi Postur")
st.sidebar.markdown("""
- Pastikan wajah dan badan bagian atas terlihat jelas di kamera.
- Pencahayaan cukup terang.
- Info postur, durasi, dan peringatan akan tampil di atas video.

© 2025 Akmal Aufa Alim

Aplikasi ini dikembangkan untuk membantu pengguna dalam memantau postur tubuh mereka secara real-time.
""")

# VideoProcessor untuk streamlit-webrtc
class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose()
        # Inisialisasi session_state jika belum ada
        if "last_posture" not in st.session_state:
            st.session_state.last_posture = None
            st.session_state.start_time = time.time()
            st.session_state.duration = 0.0
            st.session_state.warning_shown = False
        self.last_posture = st.session_state.last_posture
        self.start_time = st.session_state.start_time
        self.duration = st.session_state.duration
        self.warning_shown = st.session_state.warning_shown
        self.last_update = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        keypoints = extract_keypoints(results)
        scaled = scaler.transform([keypoints])
        pred = model.predict(scaled)
        posture = label_encoder.inverse_transform(pred)[0]

        # Debug: Print prediksi setiap 30 frame (sekitar 1 detik)
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            print(f"Debug - Keypoints sum: {np.sum(keypoints):.3f}, Predicted: {posture}")

        # Gambar landmark pada frame
        annotated = image_rgb.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # Overlay info postur
        cv2.putText(
            annotated,
            f"Postur: {posture}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )
        # Overlay durasi
        now = time.time()
        if posture == self.last_posture:
            self.duration = now - self.start_time
        else:
            self.last_posture = posture
            self.start_time = now
            self.duration = 0.0
            self.warning_shown = False
        durasi_str = str(timedelta(seconds=int(self.duration)))
        cv2.putText(
            annotated,
            f"Durasi: {durasi_str}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 128, 0),
            2,
            cv2.LINE_AA
        )
        # Overlay notifikasi jika perlu
        if (
            self.duration >= REMINDER_THRESHOLD and
            posture is not None and posture.lower() != "looks good"
        ):
            cv2.putText(
                annotated,
                "UBAH POSTUR!",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        annotated_bgr = annotated_bgr.astype(np.uint8)
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

# Streamlit-webrtc streamer
webrtc_streamer(
    key="posture-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PostureProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
)