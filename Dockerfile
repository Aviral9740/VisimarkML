FROM python:3.10-slim

WORKDIR /app

# --- System dependencies (minimal & sufficient) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libglib2.0-0 \
    libgl1 \
    curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# --- Upgrade pip ---
RUN pip install --no-cache-dir --upgrade pip

# --- Install Python deps ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy app code ---
COPY . .

# --- App dirs ---
RUN mkdir -p Attendancedir

# --- Environment ---
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 10000

# --- Healthcheck ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:${PORT:-10000}/api/health || exit 1

# --- Run server ---
CMD gunicorn app:app \
  --bind 0.0.0.0:${PORT:-10000} \
  --timeout 120 \
  --workers 2 \
  --threads 2 \
  --worker-class gthread \
  --preload
