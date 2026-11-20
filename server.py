"""
VoxLingua107 Language Identification Server
────────────────────────────────────────────
SpeechBrain ECAPA-TDNN model for 107 spoken languages.
"""
import logging
import tempfile
from pathlib import Path

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from speechbrain.inference.classifiers import EncoderClassifier

log = logging.getLogger("voxlingua107")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-5s %(message)s", datefmt="%H:%M:%S")

MODEL_DIR = "/app/model_cache"

app = FastAPI(title="VoxLingua107", description="Spoken language identification — 107 languages")

classifier = None
ALL_LABELS = []


@app.on_event("startup")
def startup():
    global classifier, ALL_LABELS
    log.info("Loading VoxLingua107 model from %s", MODEL_DIR)
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir=MODEL_DIR,
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
    suffix = Path(audio.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await audio.read())
        tmp.flush()
        try:
            result = identify_audio(tmp.name, min_confidence, top_k)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return result


@app.get("/health")
async def health():
    return {"status": "ok", "model": "voxlingua107", "languages": len(ALL_LABELS)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
