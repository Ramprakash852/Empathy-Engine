# The Empathy Engine
> Text-to-speech that hears what text *feels* - and speaks accordingly.

## Why this approach
- **j-hartmann/emotion-english-distilroberta-base** over VADER: captures nuance in mixed-emotion text (7 classes vs 3)
- **Parallel TTS**: gTTS + pyttsx3 run in concurrent threads. First result wins. gTTS ~600ms, pyttsx3 ~80ms offline
- **Intensity as a second dimension**: composite of CAPS ratio, punctuation density, and model confidence
- **Linear interpolation**: voice params lerp between calm/peak - no hardcoded if/elif chains

## Quick start
```bash
git clone 
cd empathy-engine
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# Open http://localhost:8000
```

## Try the API
```bash
# Emotion analysis
curl -X POST http://localhost:8000/analyze \
	-H 'Content-Type: application/json' \
	-d '{"text": "I just got promoted! Best day EVER!"}'

# Generate speech file
python cli.py "I am so furious right now!!"
```

## Voice parameter mapping
| Emotion  | Calm params              | Peak (intensity=1.0)     |
|----------|--------------------------|---------------------------|
| joy      | rate=1.25, pitch=+12%    | rate=1.45, pitch=+20%     |
| sadness  | rate=0.82, pitch=-8%     | rate=0.65, pitch=-15%     |
| anger    | rate=1.10, pitch=+4%     | rate=1.30, pitch=+8%      |

## Benchmarks (measured)
| Scenario         | Latency    |
|------------------|------------|
| Cold start       | ~90s (one-time model download) |
| Subsequent reqs  | ~1.2s avg  |
| Cache hit        | ~0.05s     |
