FROM python:3.11-slim

# System libraries install karein (Updated package names ke sath)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-xcb1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies copy aur install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baaki code copy karein
COPY . .

# Server start karein (Railway ke $PORT variable ke sath)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
