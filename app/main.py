"""FastAPI application for emotion analysis and speech synthesis."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"

load_dotenv(dotenv_path=env_path, override=True)

try:
	from .audio_processor import process_audio
	from .cache import get_cached, set_cached
	from .emotion_detector import detect_emotion
	from .intensity_scorer import compute_intensity
	from .tts_orchestrator import generate_tts
	from .voice_mapper import get_voice_params
except ImportError:  # Supports running as a plain script from the app directory.
	from audio_processor import process_audio
	from cache import get_cached, set_cached
	from emotion_detector import detect_emotion
	from intensity_scorer import compute_intensity
	from tts_orchestrator import generate_tts
	from voice_mapper import get_voice_params


logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
	text: str = Field(..., min_length=1, description="Input text to analyze")


class SpeakRequest(BaseModel):
	text: str = Field(..., min_length=1, description="Input text to synthesize")
	lang: str = Field(default="en", description="Language code for gTTS")
	prefer_quality: bool = Field(default=False, description="Prefer gTTS when available")
	output_bitrate: str = Field(default="192k", description="Output MP3 bitrate")


app = FastAPI(
	title="Empathy API",
	description="Analyze text emotion and generate expressive speech.",
	version="1.0.0",
)

_PROJECT_ROOT = BASE_DIR
_TEMPLATES_DIR = _PROJECT_ROOT / "templates"
_STATIC_DIR = _PROJECT_ROOT / "static"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def index(request: Request) -> Response:
	"""Serve web UI when templates exist, else return a health payload."""
	if _TEMPLATES_DIR.exists() and (_TEMPLATES_DIR / "index.html").exists():
		return templates.TemplateResponse("index.html", {"request": request})
	return JSONResponse({"status": "ok"})


@app.get("/health")
def health() -> dict[str, str]:
	"""Health check endpoint for API monitoring."""
	return {"status": "ok"}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict:
	"""Analyze text and return emotion, intensity, and voice parameters."""
	try:
		emotion_result = detect_emotion(payload.text)
		intensity = compute_intensity(payload.text, emotion_result=emotion_result)
		voice_params = get_voice_params(str(emotion_result["emotion"]), intensity)

		return {
			"text": payload.text,
			"emotion": emotion_result["emotion"],
			"confidence": emotion_result["confidence"],
			"all_scores": emotion_result.get("all_scores", {}),
			"intensity": intensity,
			"voice_params": voice_params,
			"emotion_full": emotion_result,
		}
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.post("/speak")
def speak(payload: SpeakRequest) -> Response:
	"""Generate expressive speech for input text and return MP3 audio bytes."""
	try:
		emotion_result = detect_emotion(payload.text)
		intensity = compute_intensity(payload.text, emotion_result=emotion_result)
		voice_params = get_voice_params(str(emotion_result["emotion"]), intensity)
		text = f"{payload.lang}|{payload.prefer_quality}|{payload.text}"
		emotion = str(emotion_result["emotion"])
		# cached = get_cached(...)
		cached = None
		if cached:
			if isinstance(cached, dict):
				cached_audio = cached.get("audio_bytes", b"")
				cached_engine = str(cached.get("engine", "cache"))
			else:
				# Backward compatibility for older cache entries storing raw bytes.
				cached_audio = cached
				cached_engine = "cache"
			logger.info(
				"speak_result emotion=%s intensity=%s engine=%s cache=hit",
				emotion,
				intensity,
				cached_engine,
			)

			return Response(
				content=cached_audio,
				media_type="audio/mpeg",
				headers={
					"X-TTS-Engine": cached_engine,
					"X-Emotion": emotion,
					"X-Intensity": str(intensity),
					"X-Cache": "hit",
				},
			)

		raw_audio, engine_name = generate_tts(
			text=payload.text,
			lang=payload.lang,
			prefer_quality=payload.prefer_quality,
		)
		input_format = "wav" if engine_name == "pyttsx3" else "mp3"

		processed_audio = process_audio(
			audio_bytes=raw_audio,
			voice_params={
				"rate": float(voice_params["rate"]),
				"pitch": float(voice_params["pitch"]),
				"vol": float(voice_params["vol"]),
			},
			input_format=input_format,
			output_bitrate=payload.output_bitrate,
		)

		filename = f"output_{uuid.uuid4().hex[:8]}.mp3"
		(_STATIC_DIR / filename).write_bytes(processed_audio)
		logger.info(f"Saved: static/{filename}")

		set_cached(
			text,
			emotion,
			intensity,
			{"audio_bytes": processed_audio, "engine": engine_name},
		)
		logger.info(
			"speak_result emotion=%s intensity=%s engine=%s cache=miss",
			emotion,
			intensity,
			engine_name,
		)

		return Response(
			content=processed_audio,
			media_type="audio/mpeg",
			headers={
				"X-TTS-Engine": engine_name,
				"X-Emotion": str(emotion_result["emotion"]),
				"X-Intensity": str(intensity),
				"X-Cache": "miss",
			},
		)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Speech generation failed: {exc}") from exc