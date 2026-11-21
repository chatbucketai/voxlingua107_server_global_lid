# ---- Stage 1: Build ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY auth.py server.py ./

# Model baked into the image as real files, then loaded once exactly as at
# startup, so the container never downloads or writes anything.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('speechbrain/lang-id-voxlingua107-ecapa', local_dir='/app/model_cache')" \
 && python -c "import server; server.load_model()" \
 && rm -rf /app/model_cache/.cache \
 && chmod -R a+rX /app/model_cache

ENV PYTHONUNBUFFERED=1 S2T_PORT=8000 HF_HUB_OFFLINE=1
USER 65532:65532
EXPOSE 8000

CMD ["python", "server.py"]
