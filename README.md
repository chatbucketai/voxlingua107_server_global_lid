# VoxLingua107 — Global Language Identification Server

Spoken language identification API for **107 languages** using SpeechBrain's ECAPA-TDNN model.

## Architecture

```
Audio → SpeechBrain ECAPA-TDNN (107 languages) → Language + Confidence + Top-K
```

- **Model:** `speechbrain/lang-id-voxlingua107-ecapa` (~20M params, FP32)
- **Framework:** PyTorch + torchaudio + SpeechBrain
- **Accuracy:** 95% (tested across 40 languages)
- **Inference:** ~40–100ms on CPU
- **Input:** Audio file (.wav, .mp3, .ogg), minimum 1 second

## API

### `POST /identify`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | file | required | Audio file |
| `min_confidence` | float | 0.0 | Minimum confidence threshold (0.0–1.0) |
| `top_k` | int | 5 | Number of top predictions (1–107) |

**Response:**
```json
{
  "language": "hi",
  "confidence": 0.9842,
  "duration": 3.5,
  "top_k": [
    {"language": "hi", "language_name": "Hindi", "probability": 0.9842},
    {"language": "mr", "language_name": "Marathi", "probability": 0.0091}
  ]
}
```

### `GET /health`

Returns `{"status": "ok", "model": "voxlingua107", "languages": 107}`

## Run locally

```bash
pip install -r requirements.txt
python server.py
```

Server starts on `http://localhost:8000`

## Docker

```bash
docker build -t voxlingua107 .
docker run -p 8000:8000 voxlingua107
```

Model is downloaded from HuggingFace during Docker build — container starts fast.

## Deploy (GCP Cloud Run)

```bash
gcloud builds submit --config=cloudbuild.yaml
```

**Cloud Run config:** 4 vCPU, 4Gi RAM, asia-south1, min-instances=1

## Supported languages (107)

Afrikaans, Amharic, Arabic, Assamese, Azerbaijani, Bashkir, Belarusian, Bengali, Bosnian, Breton, Bulgarian, Burmese, Catalan, Cebuano, Central Khmer, Chinese, Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Faroese, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Haitian Creole, Hausa, Hawaiian, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Interlingua, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Korean, Kyrgyz, Lao, Latin, Latvian, Lingala, Lithuanian, Luxembourgish, Macedonian, Malay, Malayalam, Maltese, Mandarin, Marathi, Mongolian, Nepali, Norwegian, Norwegian Nynorsk, Occitan, Oriya, Panjabi, Persian, Polish, Portuguese, Pushto, Romanian, Russian, Sanskrit, Scottish Gaelic, Serbian, Sindhi, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tagalog, Tajik, Tamil, Tatar, Telugu, Thai, Tibetan, Turkish, Turkmen, Ukrainian, Urdu, Uzbek, Vietnamese, Waray, Welsh, Yiddish, Yoruba
