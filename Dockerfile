# ===========================================================
# Face Attendance Flask API - DeepFace + TensorFlow + MongoDB
# Python 3.9 (Bookworm) base image for maximum compatibility
# ===========================================================

FROM python:3.9-bookworm

# Set working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy dependency list
COPY requirements.txt .

# Install dependencies from requirements
RUN pip install --no-cache-dir -r requirements.txt

# ✅ FIX: Install TensorFlow 2.13 (last stable version with built-in Keras)
# OR install separate Keras 2.x package
RUN pip install --no-cache-dir tensorflow==2.13.1 keras==2.13.1 h5py==3.10.0

# Optional: Preload DeepFace model for faster container startup
# RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512')"

# Copy the rest of your application code
COPY . .

# Create directory for storing face images
RUN mkdir -p Attendancedir

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=5000

# Expose the Flask app port
EXPOSE 5000

# Health check (optional but good practice)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:5000/api/health || exit 1

# ✅ Run app using Gunicorn (production-ready)
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2 --threads 2 --worker-class gthread --preload