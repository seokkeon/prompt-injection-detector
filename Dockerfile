FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-eng \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p logs data/raw data/malicious_samples data/benign_samples models/cache

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
