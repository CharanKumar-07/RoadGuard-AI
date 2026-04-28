FROM python:3.10-slim

WORKDIR /app

# OpenCV and GPU-friendly dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Core ML requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/      ./src/
COPY models/   ./models/
COPY config/   ./config/
RUN mkdir -p evidence

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.processor", \
    "--cameras", "config/cameras.yaml", \
    "--calibration", "config/calibration.yaml"]
