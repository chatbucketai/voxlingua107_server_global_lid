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

Server starts on `http://localhost:8000` (`S2T_PORT` to change)

## Docker

```bash
docker build -t voxlingua107 .
docker run -p 8000:8000 voxlingua107
```

Model is downloaded from HuggingFace during Docker build — container starts fast.

## Deploy (GCP Cloud Run)

Cloud Run service `stt-voxlingua107`, public at `https://stt-agent.chatbucket.chat/global_lid_voxlingua107/{identify,health}` behind the STT load balancer (Cloud Armor, managed certificate). Infra is Terraform (chat-bucket-prod-terraform, `modules/stt-cloudrun`).

`.github/workflows/gcp-build-push.yml`: every push/PR builds, smoke-tests, checks auth and scans the image. Production: run the workflow manually on `main` (target `production`). Repo variables/secrets: `PROD_*` (and `STAGE_*` for staging), same names as speech_lid.

`POST /identify` needs `Authorization: Bearer <token>`: an STT token (HS256, aud `stt`, key `S2T_TOKEN_KEY` / `S2T_TOKEN_KEY_B2B`) or a ChatBucket app login token. With no key configured (local runs) every request is allowed.

## Supported languages (107)

Afrikaans, Amharic, Arabic, Assamese, Azerbaijani, Bashkir, Belarusian, Bengali, Bosnian, Breton, Bulgarian, Burmese, Catalan, Cebuano, Central Khmer, Chinese, Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Faroese, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Haitian Creole, Hausa, Hawaiian, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Interlingua, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Korean, Kyrgyz, Lao, Latin, Latvian, Lingala, Lithuanian, Luxembourgish, Macedonian, Malay, Malayalam, Maltese, Mandarin, Marathi, Mongolian, Nepali, Norwegian, Norwegian Nynorsk, Occitan, Oriya, Panjabi, Persian, Polish, Portuguese, Pushto, Romanian, Russian, Sanskrit, Scottish Gaelic, Serbian, Sindhi, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tagalog, Tajik, Tamil, Tatar, Telugu, Thai, Tibetan, Turkish, Turkmen, Ukrainian, Urdu, Uzbek, Vietnamese, Waray, Welsh, Yiddish, Yoruba
