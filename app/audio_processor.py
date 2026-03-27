"""Audio post-processing utilities.

Applies rate, pitch, and volume transforms to input audio bytes and
returns MP3 bytes.
"""

from __future__ import annotations

import io
import math

from pydub import AudioSegment, effects
from pydub.effects import normalize


def _clamp(value: float, minimum: float, maximum: float) -> float:
	"""Clamp numeric value into [minimum, maximum]."""
	return max(minimum, min(maximum, value))


def _guess_audio_format(audio_bytes: bytes) -> str:
	"""Best-effort format detection from magic bytes."""
	if audio_bytes.startswith(b"RIFF"):
		return "wav"
	if audio_bytes.startswith(b"ID3") or audio_bytes[:2] == b"\xff\xfb":
		return "mp3"
	if audio_bytes.startswith(b"OggS"):
		return "ogg"
	return "mp3"


def _apply_pitch_shift(segment: AudioSegment, semitones: float) -> AudioSegment:
	"""Pitch shift by semitones using frame-rate resampling."""
	if abs(semitones) < 1e-6:
		return segment

	pitch_factor = 2.0 ** (semitones / 12.0)
	new_frame_rate = int(segment.frame_rate * pitch_factor)
	shifted = segment._spawn(segment.raw_data, overrides={"frame_rate": new_frame_rate})
	return shifted.set_frame_rate(segment.frame_rate)


def _apply_speed(segment: AudioSegment, rate: float) -> AudioSegment:
	"""Apply playback speed change.

	For rate > 1, use pydub's speedup effect.
	For rate < 1, use frame-rate trick as a practical fallback.
	"""
	if abs(rate - 1.0) < 1e-6:
		return segment

	if rate > 1.0:
		return effects.speedup(segment, playback_speed=rate, chunk_size=120, crossfade=20)

	slower_frame_rate = max(1000, int(segment.frame_rate * rate))
	slower = segment._spawn(segment.raw_data, overrides={"frame_rate": slower_frame_rate})
	return slower.set_frame_rate(segment.frame_rate)


def _apply_volume(segment: AudioSegment, volume: float) -> AudioSegment:
	"""Apply linear volume scale where 1.0 means unchanged."""
	if volume <= 0:
		return segment - 120

	if abs(volume - 1.0) < 1e-6:
		return segment

	gain_db = 20.0 * math.log10(volume)
	return segment.apply_gain(gain_db)


def add_pauses(text: str, audio_segment: AudioSegment) -> AudioSegment:
	"""Append silence based on punctuation in source text."""
	if "..." in text:
		silence = AudioSegment.silent(duration=400)
		audio_segment += silence
	if "." in text:
		silence = AudioSegment.silent(duration=200)
		audio_segment += silence
	return audio_segment


def process_audio(
	audio_bytes: bytes,
	text: str | None = None,
	voice_params: dict[str, float] | None = None,
	rate: float = 1.0,
	pitch: float = 0.0,
	volume: float = 1.0,
	input_format: str | None = None,
	output_bitrate: str = "192k",
	normalize_audio: bool = True,
) -> bytes:
	"""Modify audio and return MP3 bytes.

	Args:
		audio_bytes: Input audio as bytes.
		text: Optional source text used for punctuation-aware pauses.
		voice_params: Optional dict with keys like rate, pitch and vol/volume.
		rate: Playback speed multiplier (recommended 0.6 to 1.8).
		pitch: Pitch shift value. If voice_params is passed, this is treated as
			pitch-points and converted to semitones using pitch/8.0.
		volume: Linear gain multiplier (1.0 means unchanged).
		input_format: Optional input format override (e.g. mp3, wav).
		output_bitrate: MP3 bitrate (e.g. 128k, 192k).
		normalize_audio: Normalize final signal to reduce clipping risk.
	"""
	if not isinstance(audio_bytes, (bytes, bytearray)) or len(audio_bytes) == 0:
		raise ValueError("audio_bytes must be non-empty bytes")

	resolved_rate = float(rate)
	resolved_pitch = float(pitch)
	resolved_volume = float(volume)

	if voice_params is not None:
		if not isinstance(voice_params, dict):
			raise ValueError("voice_params must be a dictionary when provided")
		resolved_rate = float(1.0 + (voice_params.get("rate", 1.0) - 1.0) * 1.5)
		resolved_pitch = float(voice_params.get("pitch", resolved_pitch))
		resolved_volume = float(voice_params.get("vol", voice_params.get("volume", resolved_volume)))
		# Voice map pitch values are expressed as pitch points; convert to semitones.
		resolved_pitch = resolved_pitch / 8.0

	emotion = str(voice_params.get("emotion", "")).lower() if voice_params else ""

	effective_rate = resolved_rate
	normalized_rate = _clamp(effective_rate, 0.5, 2.0)
	normalized_pitch = _clamp(resolved_pitch, -24.0, 24.0)
	normalized_volume = _clamp(resolved_volume, 0.0, 2.0)

	print(
		"audio_processor params:",
		{
			"input_rate": round(resolved_rate, 4),
			"effective_rate": round(effective_rate, 4),
			"applied_rate": round(normalized_rate, 4),
			"applied_pitch_semitones": round(normalized_pitch, 4),
			"applied_volume": round(normalized_volume, 4),
			"emotion": emotion or "n/a",
			"input_format": input_format or "auto",
		},
	)

	source_format = input_format or _guess_audio_format(bytes(audio_bytes))
	source_buffer = io.BytesIO(bytes(audio_bytes))
	seg = AudioSegment.from_file(source_buffer, format=source_format)

	seg = _apply_pitch_shift(seg, normalized_pitch)
	seg = _apply_speed(seg, normalized_rate)
	seg = _apply_volume(seg, normalized_volume)
	if text:
		seg = add_pauses(text, seg)
	if emotion == "sadness":
		silence = AudioSegment.silent(duration=500)
		seg += silence
	if normalize_audio:
		seg = normalize(seg)

	out_buffer = io.BytesIO()
	seg.export(out_buffer, format="mp3", bitrate=output_bitrate)
	return out_buffer.getvalue()