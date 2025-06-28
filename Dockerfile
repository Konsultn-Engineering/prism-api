FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Force fresh APT cache and switch mirror to reduce hash sum mismatch
RUN echo 'Acquire::http::No-Cache "true";' > /etc/apt/apt.conf.d/99no-cache && \
    sed -i 's|http://deb.debian.org|http://ftp.de.debian.org|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y \
        git \
        gcc \
        ffmpeg \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf && \
    apt-get update && apt-get install -y iproute2 && \
    ip link set dev eth0 mtu 1400 || true

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel urllib3 && \
    pip install --no-cache-dir --timeout 120 --retries 10 \
        -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install ffmpeg-python
RUN python -m pip install python-multipart


CMD ["uvicorn", "embed_server:app", "--host", "0.0.0.0", "--port", "8000"]
