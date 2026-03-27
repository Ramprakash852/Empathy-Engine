"""Text-to-speech orchestration across ElevenLabs, gTTS, and pyttsx3."""

from __future__ import annotations

import io
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from gtts import gTTS
import pyttsx3

try:
	from dotenv import load_dotenv
except ImportError:
	load_dotenv = None

try:
	from elevenlabs.client import ElevenLabs
except ImportError:  # Optional dependency
	_elevenlabs_generate = None
	set_api_key = None

logger = logging.getLogger(__name__)
GTTS_TIMEOUT = 8


def _get_elevenlabs_api_key() -> str:
	key = os.getenv("ELEVENLABS_API_KEY", "").strip()
	print("DEBUG KEY IN ORCHESTRATOR:", key)
	return key


def _is_elevenlabs_available() -> bool:
    try:
        from elevenlabs.client import ElevenLabs
        return bool(os.getenv("ELEVENLABS_API_KEY"))
    except ImportError:
        return False


def elevenlabs_generate(text: str) -> bytes:
    from elevenlabs.client import ElevenLabs
    import os

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ELEVENLABS_API_KEY")

    client = ElevenLabs(api_key=api_key)

    audio = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # "George" voice ID, can be changed to any available voice  
        model_id="eleven_turbo_v2"  # ✅ FIXED
    )

    return b"".join(audio)


def _synthesize_with_gtts(text: str, lang: str = "en") -> bytes:
	"""Synthesize MP3 bytes with gTTS."""
	buffer = io.BytesIO()
	tts = gTTS(text=text, lang=lang)
	tts.write_to_fp(buffer)
	return buffer.getvalue()


def _synthesize_with_pyttsx3(text: str) -> bytes:
	"""Synthesize WAV bytes with pyttsx3 via a temporary file."""
	engine = pyttsx3.init()
	temp_path: Path | None = None
	try:
		with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
			temp_path = Path(temp_file.name)

		engine.save_to_file(text, str(temp_path))
		engine.runAndWait()

		if not temp_path.exists() or temp_path.stat().st_size == 0:
			raise RuntimeError("pyttsx3 produced an empty audio file")

		return temp_path.read_bytes()
	finally:
		try:
			engine.stop()
		except Exception:
			pass
		if temp_path and temp_path.exists():
			temp_path.unlink(missing_ok=True)


def generate_tts(
	text: str,
	lang: str = "en",
	prefer_quality: bool = False,
	gtts_timeout: int = GTTS_TIMEOUT,
) -> tuple[bytes, str]:
	"""Generate speech in parallel and return (audio_bytes, engine_name).

	Engine priority is always: elevenlabs, gtts, pyttsx3.
	"""
	if not isinstance(text, str) or not text.strip():
		raise ValueError("text must be a non-empty string")

	clean_text = text.strip()
	results: dict[str, bytes] = {}
	failures: dict[str, str] = {}

	with ThreadPoolExecutor(max_workers=3) as executor:
		logger.info("gtts started")
		gtts_future = executor.submit(_synthesize_with_gtts, clean_text, lang)
		logger.info("pyttsx3 started")
		pyttsx3_future = executor.submit(_synthesize_with_pyttsx3, clean_text)

		futures: dict[Any, str] = {
			gtts_future: "gtts",
			pyttsx3_future: "pyttsx3",
		}
		if _is_elevenlabs_available():
			logger.info("elevenlabs started")
			futures[executor.submit(elevenlabs_generate, clean_text)] = "elevenlabs"
		else:
			logger.warning("elevenlabs unavailable: missing package or ELEVENLABS_API_KEY")

		max_wait = max(1, int(gtts_timeout))
		done, not_done = wait(set(futures.keys()), timeout=max_wait)
		if not_done:
			logger.warning("TTS orchestration timed out after %s seconds", max_wait)

		for future in done:
			engine_name = futures[future]
			try:
				audio_bytes = future.result()
				if not isinstance(audio_bytes, (bytes, bytearray)) or len(audio_bytes) == 0:
					raise RuntimeError("engine returned empty audio")

				results[engine_name] = bytes(audio_bytes)
				logger.info("%s succeeded", engine_name)
			except Exception as exc:
				failures[engine_name] = str(exc)
				logger.warning("%s failed: %s", engine_name, exc)

		for future in not_done:
			engine_name = futures[future]
			failures[engine_name] = "timeout"
			logger.warning("%s failed: timeout", engine_name)

	completed_engines = list(results.keys())
	print(f"TTS completed engines: {completed_engines}")

	if "elevenlabs" in results:
		selected_engine = "elevenlabs"
	elif "gtts" in results:
		selected_engine = "gtts"
	elif "pyttsx3" in results:
		selected_engine = "pyttsx3"
	else:
		raise RuntimeError("All TTS engines failed")

	priority_order = ("elevenlabs", "gtts", "pyttsx3")

	for higher_engine in priority_order:
		if higher_engine == selected_engine:
			break
		if higher_engine in failures:
			logger.warning(
				"Primary engine failed (%s: %s); fallback succeeded (%s)",
				higher_engine,
				failures[higher_engine],
				selected_engine,
			)

	logger.info("selected_tts_engine=%s", selected_engine)
	print(f"TTS selected engine: {selected_engine}")

	return results[selected_engine], selected_engine


def generate_speech(
	text: str,
	lang: str = "en",
	prefer_quality: bool = False,
	gtts_timeout: int = GTTS_TIMEOUT,
) -> dict[str, Any]:
	"""Compatibility wrapper that returns dict output."""
	audio_bytes, engine_name = generate_tts(
		text=text,
		lang=lang,
		prefer_quality=prefer_quality,
		gtts_timeout=gtts_timeout,
	)
	audio_format = "wav" if engine_name == "pyttsx3" else "mp3"
	return {
		"audio_bytes": audio_bytes,
		"engine": engine_name,
		"format": audio_format,
	}