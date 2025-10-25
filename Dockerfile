# ===========================================================
# Face Attendance Flask API - DeepFace + TensorFlow + MongoDB
# Optimized for Render / Python 3.9
# ===========================================================
FROM python:3.9-bookworm
# ---------- System Setup ----------
WORKDIR /app

# Install required system packages for OpenCV, TensorFlow, DeepFace
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ---------- Python Dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# (Optional) Preload the DeepFace model to speed up first request
# RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512')"

# ---------- App Setup ----------
COPY . .

# Create directory for stored attendance images
RUN mkdir -p Attendancedir

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=5000

# Expose service port
EXPOSE 5000

# ---------- Health Check ----------
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:5000/api/health || exit 1

# ---------- Run ----------
# --preload loads DeepFace once globally (faster startup)
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2 --threads 2 --worker-class gthread --preload
