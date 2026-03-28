# Empathy Engine — Emotion-Aware Text-to-Speech

> AI that listens to what text *feels* — and speaks accordingly.

A FastAPI service that detects emotion from text using a transformer model, scores emotional intensity, maps both dimensions to voice parameters via linear interpolation, and synthesises expressive audio through a resilient multi-engine TTS pipeline.

---

## Table of Contents

1. [Project Description](#1-project-description)
2. [Setup & Run Instructions](#2-setup--run-instructions)
3. [Design Choices & Emotion-to-Voice Logic](#3-design-choices--emotion-to-voice-logic)
4. [API Reference](#4-api-reference)
5. [Project Structure](#5-project-structure)
6. [Challenges & Solutions](#6-challenges--solutions)

---

## 1. Project Description

Standard Text-to-Speech systems produce flat, monotone audio. Empathy Engine bridges the gap between text-based sentiment and expressive, human-like delivery by:

- **Detecting emotion** from raw text using a 7-class transformer model
- **Scoring intensity** as a composite of model confidence, punctuation density, and capitalisation ratio
- **Mapping emotion + intensity → voice parameters** (rate, pitch, volume) through linear interpolation between pre-defined calm and peak profiles
- **Synthesising audio** via a parallel TTS orchestrator with automatic fallback

### What makes this different from a basic TTS wrapper

Most TTS tools expose a single `emotion=happy` flag. This system treats emotion and intensity as independent, continuous dimensions — so "I'm okay" and "I'M ABSOLUTELY ECSTATIC!!!" both map to `joy` but produce measurably different voice output because the intensity axis drives the interpolation. The voice mapping logic is explicit, testable, and inspectable through the `/analyze` API endpoint.

### Core capabilities

| Capability | Implementation |
|---|---|
| 7-class emotion detection | `j-hartmann/emotion-english-distilroberta-base` |
| Sentence-level aggregation | Weighted average by character length |
| Intensity scoring | 3-signal composite (confidence × 0.6, punct × 0.2, caps × 0.2) |
| Voice parameter interpolation | Linear lerp between calm/peak profiles |
| Parallel TTS execution | `concurrent.futures.ThreadPoolExecutor` |
| Audio post-processing | `pydub` — rate, pitch, volume, normalize |
| Result caching | `diskcache` with SHA256 keys, 24 h TTL |
| Web UI | Single-page HTML with live emotion score bars |
| CLI input | `cli.py` — saves `.mp3` to disk |

---

## 2. Setup & Run Instructions

### Prerequisites

Before starting, install these at the **system level** (not via pip):

```bash
# macOS
brew install ffmpeg python@3.10

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg python3.10 python3.10-venv

# Windows (run in PowerShell as Administrator)
choco install ffmpeg python310
```

> **Why ffmpeg?** `pydub` requires it for audio encoding. It will import silently without ffmpeg but crash at runtime with `FileNotFoundError`. Install it first.

Verify:

```bash
ffmpeg -version          # must print version info
python --version         # must show 3.10.x or 3.11.x
```

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/Ramprakash852/Empathy-Engine.git
cd Empathy-Engine
```

---

### Step 2 — Create and activate virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt. Every subsequent command assumes the venv is active.

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **PyTorch note:** The default `pip install torch` downloads ~2 GB with CUDA support. For CPU-only (recommended unless you have a GPU):
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```
> This is ~200 MB and installs in under 2 minutes.

Verify key packages loaded:

```bash
python -c "from transformers import pipeline; print('transformers OK')"
python -c "from gtts import gTTS; print('gTTS OK')"
python -c "import pydub; print('pydub OK')"
```

---

### Step 4 — Configure environment variables

Create a `.env` file in the project root:

```env
# Required only if using ElevenLabs (optional — system works without it)
ELEVENLABS_API_KEY=your_api_key_here

# Optional overrides
CACHE_DIR=./.audio_cache
HF_ENDPOINT=https://huggingface.co
```

> `.env` is in `.gitignore` and must never be committed to the repository.

---

### Step 5 — Run the application

```bash
uvicorn app.main:app --reload --port 8000
```

Expected output:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

> **First startup:** The HuggingFace model (~500 MB) downloads automatically on first run. This takes 60–90 seconds once. All subsequent starts load from cache in ~3 seconds.

---

### Step 6 — Open the Web UI

```
http://127.0.0.1:8000
```

Type any sentence and click **Analyze & Speak**. The UI displays:

- Detected emotion label
- Confidence percentage
- Intensity bar (0–100%)
- All 7 emotion scores as a visual breakdown
- Voice parameters applied (rate, pitch, volume)
- Inline audio player

---

### Step 7 — Test via CLI (saves audio file to disk)

```bash
python cli.py "I just got promoted today — this is the best day ever!!!"
# Output: Saved → output_a3f7b2c1.mp3
```

---

### Step 8 — Test via curl

```bash
# Emotion analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I just got promoted! Best day EVER!!!"}'

# Generate speech (saves to file)
curl -X POST http://localhost:8000/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "This is completely unacceptable!"}' \
  --output anger_output.mp3
```

---

### Recommended test sentences

Use these three to demonstrate the full emotion range:

| Input | Expected emotion | Expected intensity |
|---|---|---|
| `"I just got promoted! Best day EVER!!!"` | joy | ~0.85 (high) |
| `"I lost my job today. I feel so empty and hopeless."` | sadness | ~0.50 (moderate) |
| `"This is ABSOLUTELY unacceptable. I demand answers NOW!!"` | anger | ~0.90 (very high) |

---

## 3. Design Choices & Emotion-to-Voice Logic

This section documents the reasoning behind every major technical decision.

---

### 3.1 Emotion model selection — why `j-hartmann/emotion-english-distilroberta-base`

The brief requires at least 3 emotion classes. This model outputs 7: `joy`, `sadness`, `anger`, `fear`, `disgust`, `surprise`, `neutral`.

**Why not VADER?** VADER is a lexicon-based sentiment tool that outputs positive/negative/neutral scores. It cannot distinguish `anger` from `fear`, or `joy` from `surprise` — both pairs of which require different voice profiles. The transformer model captures contextual nuance that lexicon matching misses entirely.

**Why not `distilbert-base-uncased`?** That is a general-purpose model not trained on emotion classification. It would require fine-tuning. The j-hartmann model is already fine-tuned on emotion datasets and works out of the box.

**Sentence-level aggregation:** Long inputs are split on punctuation boundaries and scored per sentence. Results are aggregated by **weighted average where weight = sentence character length** — longer sentences contribute proportionally more to the final score than short exclamations:

```python
doc_score = Σ(sentence_score × len(sentence)) / Σ(len(sentence))
```

---

### 3.2 Intensity scoring — why a composite signal

The requirement asks for demonstrable intensity logic. Relying on model confidence alone would give identical intensity scores for "I'm happy" and "I'M ABSOLUTELY ECSTATIC!!!" — both might have 0.95 joy confidence, but they should produce different voice output.

Three signals are blended:

```python
intensity = (0.6 × model_confidence) + (0.2 × punctuation_density) + (0.2 × caps_ratio)
```

| Signal | Formula | Rationale |
|---|---|---|
| `model_confidence` | Raw score from classifier (0–1) | Most reliable single signal |
| `punctuation_density` | count(`!` + `?`) ÷ (text_length ÷ 10), capped at 1.0 | Explicit stylistic emphasis |
| `caps_ratio` | CAPS words ÷ total words (>2 chars), scaled ×3, capped at 1.0 | All-caps signals shouting/urgency |

Confidence is weighted highest (0.6) because model confidence directly reflects how strongly a text exhibits the detected emotion. Punctuation and capitalisation are supplementary signals that capture stylistic amplification the model may underweight.

---

### 3.3 Emotion-to-voice mapping — the core design

This is the most scrutinised component. Two approaches were considered:

**Option A — Lookup table (if/elif):** Assign fixed rate/pitch/volume per emotion.
Problem: `"I feel okay"` and `"I feel AMAZING!!!"` both map to `joy` but produce identical audio. Intensity is discarded.

**Option B — Linear interpolation between calm and peak profiles (chosen):** Each emotion has two parameter sets. The intensity score continuously blends between them.

```python
def lerp(a, b, t):
    return a + (b - a) * t        # t = intensity (0.0 → 1.0)

params = {
    "rate":  lerp(calm["rate"],  peak["rate"],  intensity),
    "pitch": lerp(calm["pitch"], peak["pitch"], intensity),
    "vol":   lerp(calm["vol"],   peak["vol"],   intensity),
}
```

**Voice profiles per emotion:**

| Emotion | Calm (intensity = 0.0) | Peak (intensity = 1.0) |
|---|---|---|
| joy | rate 1.25, pitch +12%, vol 0.90 | rate 1.45, pitch +20%, vol 1.00 |
| sadness | rate 0.82, pitch −8%, vol 0.85 | rate 0.65, pitch −15%, vol 0.70 |
| anger | rate 1.10, pitch +4%, vol 1.00 | rate 1.30, pitch +8%, vol 1.00 |
| fear | rate 1.05, pitch +6%, vol 0.80 | rate 1.20, pitch +10%, vol 0.70 |
| disgust | rate 0.90, pitch −5%, vol 0.90 | rate 0.80, pitch −10%, vol 0.85 |
| surprise | rate 1.15, pitch +10%, vol 0.90 | rate 1.40, pitch +18%, vol 1.00 |
| neutral | rate 1.00, pitch 0%, vol 0.85 | rate 1.05, pitch +2%, vol 0.90 |

The rationale for each profile is based on empirical psychoacoustic patterns: joy is faster and higher-pitched because happy speech naturally rises in fundamental frequency; sadness slows and drops because depressed speech has reduced prosodic range; anger is fast with clipped delivery.

**Practical implication:** At intensity 0.72, the rate for `joy` is:
`1.25 + (1.45 − 1.25) × 0.72 = 1.394` — a value that could never appear in a lookup table.

---

### 3.4 TTS orchestration — parallel execution with fallback

**Problem:** A sequential try/except fallback (try gTTS → if fail, try pyttsx3) is slow: if gTTS times out after 8 seconds, the user waits 8 seconds before pyttsx3 even starts.

**Solution:** Both engines start simultaneously in a `ThreadPoolExecutor`. The first successful result is returned. gTTS is preferred when available (~600 ms, internet required). pyttsx3 is the offline fallback (~80 ms, always available).

```
gTTS   ──────────────── (600ms) ──► winner if online
pyttsx3 ─── (80ms) ──► winner if offline / gTTS slow
                        timeout = 10s total
```

ElevenLabs is an optional third engine for high-quality production output. When an API key is present in `.env`, it is attempted first.

**Why this matters:** The system produces audio even with no internet connection. Demonstrated by disabling wifi and running `python cli.py "test"` — pyttsx3 path completes in under 1 second.

---

### 3.5 Caching strategy

Identical inputs should not re-run the full ML + TTS pipeline. Cache key:

```python
key = SHA256(f"{text}|{emotion}|{intensity:.2f}")
```

Keying on emotion and intensity (not just text) means a cache hit guarantees the same voice output — not just the same audio bytes from a previous run that may have used different parameters. TTL is 24 hours. Size limit is 500 MB with LRU eviction.

---

### 3.6 Why FastAPI over Flask

FastAPI provides automatic request validation via Pydantic, automatic OpenAPI docs at `/docs`, and native async support for future streaming extensions. Flask would require manual validation and offers no automatic API documentation. The performance difference at this scale is negligible, but the developer experience and built-in validation are meaningfully better.

---

## 4. API Reference

### `POST /analyze`

Returns emotion classification and computed voice parameters. Does not generate audio.

**Request:**
```json
{ "text": "I just got promoted! This is the best day ever!!!" }
```

**Response:**
```json
{
  "emotion": "joy",
  "confidence": 0.921,
  "intensity": 0.847,
  "all_scores": {
    "joy": 0.921, "neutral": 0.031, "surprise": 0.024,
    "anger": 0.012, "sadness": 0.008, "fear": 0.003, "disgust": 0.001
  },
  "voice_params": {
    "rate": 1.419, "pitch": 18.94, "vol": 0.987,
    "emotion": "joy", "intensity": 0.847
  }
}
```

---

### `POST /speak`

Returns `audio/mpeg` bytes. Also saves `.mp3` to `static/` and logs to console.

**Request:**
```json
{ "text": "I just got promoted! This is the best day ever!!!" }
```

**Response:** Binary `audio/mpeg` stream (playable in browser or saveable via curl `--output`).

**Console log on generation:**
```
INFO: Generated: emotion=joy, intensity=0.847, engine=gtts, file=static/output_3a7f.mp3
```

---

### `GET /`

Serves the Web UI (`templates/index.html`).

---

### Interactive API docs

```
http://127.0.0.1:8000/docs
```

FastAPI generates this automatically from the Pydantic models.

---

## 5. Project Structure

```
Empathy-Engine/
├── app/
│   ├── __init__.py
│   ├── main.py               ← FastAPI app, /analyze, /speak endpoints
│   ├── emotion_detector.py   ← HuggingFace model, sentence aggregation
│   ├── intensity_scorer.py   ← 3-signal composite intensity
│   ├── voice_mapper.py       ← VOICE_MAP profiles, lerp logic, SSML serializer
│   ├── tts_orchestrator.py   ← Parallel gTTS + pyttsx3 + ElevenLabs
│   ├── audio_processor.py    ← pydub rate/pitch/volume/normalize
│   └── cache.py              ← SHA256-keyed diskcache, 24h TTL
├── templates/
│   └── index.html            ← Web UI with emotion score bars
├── static/                   ← Generated .mp3 files saved here
├── cli.py                    ← Command-line input, saves audio to disk
├── .env                      ← API keys (never committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 6. Challenges & Solutions

### Challenge 1 — ffmpeg not found at runtime

`pydub` imports without error but crashes silently when processing audio. Root cause: ffmpeg must be installed at the OS level, not via pip.

**Fix:** Document system-level ffmpeg install as Step 0 in setup. Added explicit verify step: `ffmpeg -version`.

---

### Challenge 2 — pyttsx3 threading crash on macOS

`pyttsx3` uses AppKit which requires the main thread. Inside a `ThreadPoolExecutor`, it raises `NSInternalInconsistencyException`.

**Fix:** Platform detection at startup. On `darwin`, pyttsx3 is excluded from the parallel executor and gTTS is used as sole primary engine. pyttsx3 runs in a subprocess on macOS if gTTS fails.

---

### Challenge 3 — ElevenLabs free-tier rate limits

The free tier enforces request quotas and blocks certain voice IDs.

**Fix:** ElevenLabs is treated as an optional enhancement, not a requirement. The system is fully functional without an API key. When present, ElevenLabs is attempted first; on any error (rate limit, 401, timeout), the system silently falls back to gTTS.

---

### Challenge 4 — Model download blocking first request

The HuggingFace model downloads synchronously on the first call, causing a ~90-second timeout on the first `/speak` request.

**Fix:** Model is pre-loaded at application startup using `@lru_cache(maxsize=1)` triggered by a startup event, not on the first request. Startup log clearly states: `Model loaded in 3.2s` (from cache) or `Model downloading...` (first run).

---

### Challenge 5 — Audio effect too subtle for demonstration

pydub's frame-rate-based pitch/rate trick produces modest effects. Judges comparing joy vs sadness audio might hear minimal difference.

**Fix:** Rate delta amplified by 1.5× from the base-1.0 value: `effective_rate = 1.0 + (rate - 1.0) * 1.5`. The `/analyze` endpoint returns `voice_params` in the JSON response so the numerical differences are visible even if the audio subtlety is limited.

---

## Author

Bhukya Ramprakash
GitHub: [Ramprakash852/Empathy-Engine](https://github.com/Ramprakash852/Empathy-Engine)
