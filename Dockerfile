FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Only enable CPU-friendly engines on free HuggingFace Space
ENV ENABLED_ENGINES=kokoro,f5
ENV HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip \
    && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install "kokoro>=0.9.4" "f5-tts>=1.1.0" \
    && pip install fastapi "uvicorn[standard]" python-multipart soundfile numpy huggingface_hub

EXPOSE 7860
# --no-preload: don't load Qwen3 at startup (not available on CPU)
CMD ["python3", "server.py", "--host", "0.0.0.0", "--no-preload"]
