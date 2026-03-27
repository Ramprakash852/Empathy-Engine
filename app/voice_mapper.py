"""Voice parameter mapping utilities.

Maps detected emotion + intensity into speech parameters using
linear interpolation (lerp).
"""

from __future__ import annotations

from typing import Any


def _clamp(value: float, minimum: float, maximum: float) -> float:
	"""Clamp a value to the provided range."""
	return max(minimum, min(maximum, value))


def _lerp(start: float, end: float, t: float) -> float:
	"""Linear interpolation between start and end."""
	return start + (end - start) * t


# Each emotion has calm and peak parameter sets. Output is interpolated by intensity.
VOICE_MAP: dict[str, dict[str, dict[str, float]]] = {
	"joy": {
		"calm": {"rate": 1.25, "pitch": 12.0, "vol": 0.9},
		"peak": {"rate": 1.45, "pitch": 20.0, "vol": 1.0},
	},
	"sadness": {
		"calm": {"rate": 0.82, "pitch": -8.0, "vol": 0.85},
		"peak": {"rate": 0.65, "pitch": -15.0, "vol": 0.7},
	},
	"anger": {
		"calm": {"rate": 1.10, "pitch": 4.0, "vol": 1.0},
		"peak": {"rate": 1.30, "pitch": 8.0, "vol": 1.0},
	},
	"fear": {
		"calm": {"rate": 1.05, "pitch": 6.0, "vol": 0.8},
		"peak": {"rate": 1.20, "pitch": 10.0, "vol": 0.7},
	},
	"disgust": {
		"calm": {"rate": 0.90, "pitch": -5.0, "vol": 0.9},
		"peak": {"rate": 0.80, "pitch": -10.0, "vol": 0.85},
	},
	"surprise": {
		"calm": {"rate": 1.15, "pitch": 10.0, "vol": 0.9},
		"peak": {"rate": 1.40, "pitch": 18.0, "vol": 1.0},
	},
	"neutral": {
		"calm": {"rate": 1.00, "pitch": 0.0, "vol": 0.85},
		"peak": {"rate": 1.05, "pitch": 2.0, "vol": 0.9},
	},
}


def get_voice_params(emotion: str, intensity: float) -> dict[str, float | str]:
	"""Return interpolated parameters for a given emotion and intensity."""
	if not isinstance(emotion, str):
		raise ValueError("emotion must be a string")

	key = emotion.strip().lower()
	profile = VOICE_MAP.get(key, VOICE_MAP["neutral"])
	calm = profile["calm"]
	t = _clamp(float(intensity), 0.0, 1.0)

	base_rate = float(calm["rate"])
	base_pitch = float(calm["pitch"])
	base_vol = float(calm["vol"])

	rate = base_rate + (t * 0.3)
	pitch = base_pitch + (t * 10)
	volume = base_vol + (t * 0.2)

	return {
		"rate": round(rate, 4),
		"pitch": round(pitch, 4),
		"vol": round(_clamp(volume, 0.0, 1.0), 4),
		"emotion": key,
		"intensity": round(t, 4),
	}


def map_voice(emotion: str, intensity: float) -> dict[str, float]:
	"""Compatibility wrapper that returns rate, pitch, and volume keys."""
	params = get_voice_params(emotion, intensity)
	return {
		"rate": float(params["rate"]),
		"pitch": float(params["pitch"]),
		"volume": float(params["vol"]),
	}


def map_voice_from_result(emotion_result: dict[str, Any], intensity: float) -> dict[str, float]:
	"""Convenience wrapper that reads the emotion label from detector output."""
	if not isinstance(emotion_result, dict):
		raise ValueError("emotion_result must be a dictionary")
	emotion = str(emotion_result.get("emotion", "neutral"))
	return map_voice(emotion, intensity)