"""Intensity scoring utilities.

Combines simple text signals (capitalization and punctuation) with
model confidence to estimate emotional intensity on a 0..1 scale.
"""

from __future__ import annotations

import re
from typing import Any


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
	"""Clamp a numeric value into the requested range."""
	return max(minimum, min(maximum, value))


def score_intensity(text: str, model_confidence: float) -> float:
	"""Composite intensity: 0.0 (calm/neutral) to 1.0 (extreme).

	Three signals, each capped at 1.0, then blended.
	"""
	if not isinstance(text, str):
		raise ValueError("text must be a string")

	# Signal 1: CAPS ratio (exclude very short words like 'I').
	words = text.split()
	long_words = [word for word in words if len(word) > 2]
	caps_ratio = sum(1 for word in long_words if word.isupper()) / max(len(long_words), 1)
	caps = min(caps_ratio * 3.0, 1.0)

	# Signal 2: Punctuation density using ! and ?
	exclamations = len(re.findall(r"!", text))
	questions = len(re.findall(r"\?", text))
	punct_ratio = (exclamations + questions) / max(len(text) / 10, 1)
	punct = min(punct_ratio, 1.0)

	# Signal 3: Model confidence (expected in 0..1).
	confidence = _clamp(float(model_confidence))

	intensity = (0.5 * confidence) + (0.35 * punct) + (0.15 * caps)

	# Boost expressive signals
	if "!" in text:
		intensity += 0.1
	if text.isupper():
		intensity += 0.1

	return round(_clamp(intensity), 3)


def compute_intensity(
	text: str,
	model_confidence: float | None = None,
	emotion_result: dict[str, Any] | None = None,
) -> float:
	"""Compatibility wrapper around score_intensity.

	Args:
		text: Input user text.
		model_confidence: Optional confidence from detector output.
		emotion_result: Optional full detector result; if provided and
			`model_confidence` is None, uses `emotion_result['confidence']`.

	Returns:
		A float in [0, 1] representing emotional intensity.
	"""
	if not isinstance(text, str):
		raise ValueError("text must be a string")

	if model_confidence is None and isinstance(emotion_result, dict):
		confidence_value = emotion_result.get("confidence")
		if confidence_value is not None:
			try:
				model_confidence = float(confidence_value)
			except (TypeError, ValueError):
				model_confidence = None

	confidence = float(model_confidence) if model_confidence is not None else 0.5
	return score_intensity(text, confidence)