"""
VoxLingua107 Language Identification Server
────────────────────────────────────────────
SpeechBrain ECAPA-TDNN model for 107 spoken languages.
"""
import asyncio
import logging
import os
import tempfile
from pathlib import Path

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from auth import token_valid

log = logging.getLogger("voxlingua107")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-5s %(message)s", datefmt="%H:%M:%S")

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model_cache")
PORT = int(os.environ.get("S2T_PORT", "8000"))
MAX_UPLOAD_BYTES = int(os.environ.get("S2T_MAX_UPLOAD_MB", "30")) * 1024 * 1024
CORS_ORIGIN_REGEX = os.environ.get("S2T_CORS_ORIGIN_REGEX", "")

app = FastAPI(title="VoxLingua107", docs_url=None, redoc_url=None, openapi_url=None)


# Before the body is read; registered before CORS so browsers can read the 401.
@app.middleware("http")
async def require_token(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/identify":
        if not token_valid(request.headers.get("authorization", "").removeprefix("Bearer ")):
            return JSONResponse({"detail": "missing or invalid token"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


if CORS_ORIGIN_REGEX:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=CORS_ORIGIN_REGEX,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        max_age=600,
    )

classifier = None
ALL_LABELS = []


@app.on_event("startup")
def startup():
    global classifier, ALL_LABELS
    log.info("Loading VoxLingua107 model from %s", MODEL_DIR)
    classifier = EncoderClassifier.from_hparams(
        source=MODEL_DIR,  # baked into the image; nothing is downloaded or written
        savedir=MODEL_DIR,
        local_strategy=LocalStrategy.COPY_SKIP_CACHE,
    )
    ALL_LABELS = sorted(classifier.hparams.label_encoder.lab2ind.keys())
    log.info("Model loaded. %d languages available.", len(ALL_LABELS))


def identify_audio(audio_path: str, min_confidence: float = 0.0, top_k: int = 5):
    signal = classifier.load_audio(audio_path)
    duration = signal.shape[0] / 16000

    if duration < 1.0:
        raise ValueError(f"Audio too short: {duration:.1f}s (minimum 1s)")

    prediction = classifier.classify_batch(signal.unsqueeze(0))
    log_posteriors = prediction[0].squeeze()
    posteriors = torch.exp(log_posteriors)
    scores, indices = posteriors.topk(min(top_k, len(ALL_LABELS)))

    results = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        label = classifier.hparams.label_encoder.decode_ndim(idx)
        lang_code = label.split(":")[0].strip()
        lang_name = label.split(":", 1)[1].strip() if ":" in label else label
        results.append({
            "language": lang_code,
            "language_name": lang_name,
            "probability": round(score, 6),
        })

    top = results[0] if results else {"language": "unknown", "probability": 0.0}
    detected = top["language"] if top["probability"] >= min_confidence else "unknown"
    confidence = top["probability"] if detected != "unknown" else top["probability"]

    return {
        "language": detected,
        "confidence": confidence,
        "duration": round(duration, 3),
        "top_k": results,
    }


@app.post("/identify")
async def identify(
    audio: UploadFile = File(...),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    top_k: int = Query(5, ge=1, le=107),
):
    raw = await audio.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    suffix = Path(audio.filename or "audio.wav").suffix[:10]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        try:
            result = await asyncio.to_thread(identify_audio, tmp.name, min_confidence, top_k)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:
            log.exception("identify failed")
            raise HTTPException(status_code=500, detail="could not process audio")

    return result


@app.get("/health")
async def health():
    return {"status": "ok", "model": "voxlingua107", "languages": len(ALL_LABELS)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
