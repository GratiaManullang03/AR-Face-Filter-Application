FROM python:3.11-slim

# Avoid Python writing .pyc files and keep stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DISPLAY=:0

WORKDIR /app

# Install system dependencies for OpenCV, MediaPipe, GUI display, and wget
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    x11-apps \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application source
COPY src ./src
COPY main.py .
COPY assets ./assets
COPY download_models.sh .

# Create directories for models, captures, and MediaPipe cache
RUN mkdir -p models captures /.mediapipe && \
    chmod 777 /.mediapipe && \
    chmod +x download_models.sh

# Download MediaPipe models (Face Landmarker + Hand Landmarker)
RUN ./download_models.sh

CMD ["python", "main.py"]
