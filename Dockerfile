FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Pre-copy for caching
COPY requirements.txt .

# Install Python deps
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code
COPY embed_server.py .

CMD ["uvicorn", "embed_server:app", "--host", "0.0.0.0", "--port", "8000"]