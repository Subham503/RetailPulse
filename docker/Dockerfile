# RetailPulse — Multi-stage Dockerfile
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create output dirs
RUN mkdir -p data/raw data/processed data/features models monitoring/reports mlflow

# Expose ports
EXPOSE 8000 8501

# Default: start API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]