FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Override in Space Settings → Variables: kokoro,f5 (CPU) or all (GPU)
ENV ENABLED_ENGINES=all
ENV HOME=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# torch pinned: f5-tts pulls torchcodec, and an unpinned torch drifts to a release whose
# torchcodec wheel no longer loads on CPU (F5 then fails with "Could not load libtorchcodec")
RUN pip install --upgrade pip \
    && pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install "kokoro>=0.9.4" "f5-tts>=1.1.0" "torchcodec==0.7.*" \
    && pip install fastapi "uvicorn[standard]" python-multipart soundfile numpy huggingface_hub

EXPOSE 7860
# --no-preload: don't load Qwen3 at startup (not available on CPU)
CMD ["python3", "server.py", "--host", "0.0.0.0", "--no-preload"]
