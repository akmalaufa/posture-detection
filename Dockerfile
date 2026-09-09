FROM python:3.11-slim

# Mencegah Python menulis file .pyc ke disk dan mematikan buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependensi sistem operasi (wajib untuk OpenCV dan WebRTC)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir streamlit opencv-python-headless numpy==2.2.6 joblib==1.5.1 scikit-learn==1.6.1 streamlit-webrtc
# Force install mediapipe without dependency checks because it strictly wants numpy<2 but works fine with numpy 2
RUN pip install --no-cache-dir --no-deps mediapipe==0.10.21
# Install mediapipe internal dependencies manually
RUN pip install --no-cache-dir absl-py attrs flatbuffers jax jaxlib matplotlib protobuf==4.25.8 sentencepiece sounddevice

# Copy seluruh source code
COPY . .

# Expose port standar Streamlit
EXPOSE 8501

# Command untuk menjalankan aplikasi versi WebRTC
CMD ["streamlit", "run", "app_webrtc.py", "--server.port=8501", "--server.address=0.0.0.0"]
