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
COPY server.py ./

# Download model at build time so container starts fast
RUN python -c "\
from speechbrain.inference.classifiers import EncoderClassifier; \
EncoderClassifier.from_hparams( \
    source='speechbrain/lang-id-voxlingua107-ecapa', \
    savedir='/app/model_cache' \
)"

EXPOSE 8000

CMD ["python", "server.py"]
