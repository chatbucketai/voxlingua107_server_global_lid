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

# Model baked into the image as real files (no Hugging Face cache symlinks),
# so the container never downloads at startup.
RUN python -c "\
from speechbrain.inference.classifiers import EncoderClassifier; \
from speechbrain.utils.fetching import LocalStrategy; \
EncoderClassifier.from_hparams( \
    source='speechbrain/lang-id-voxlingua107-ecapa', \
    savedir='/app/model_cache', \
    local_strategy=LocalStrategy.COPY_SKIP_CACHE, \
)" && chmod -R a+rX /app/model_cache

COPY auth.py server.py ./

ENV PYTHONUNBUFFERED=1 S2T_PORT=8000 HF_HUB_OFFLINE=1
USER 65532:65532
EXPOSE 8000

CMD ["python", "server.py"]
