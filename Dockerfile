FROM python:3.11-slim

# System libraries install karein (OpenCV aur MediaPipe ke liye)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
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

# Server start karein
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
