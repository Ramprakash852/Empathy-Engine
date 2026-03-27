"""Emotion detection utilities using a Hugging Face transformer model."""

import re
from functools import lru_cache
from typing import Any

from transformers import pipeline

MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"


@lru_cache(maxsize=1)
def _get_pipeline() -> Any:
	"""Create and cache the emotion classifier pipeline."""
	return pipeline(
		task="text-classification",
		model=MODEL_ID,
		top_k=None,
		device=-1,
	)


def _split_sentences(text: str) -> list[str]:
	"""Split on punctuation boundaries and keep meaningful sentences."""
	parts = re.split(r"(?<=[.!?])\s+", text.strip())
	return [part for part in parts if len(part.strip()) > 3]


def _aggregate(sentence_results: list[tuple[list[dict[str, Any]], int]]) -> dict[str, Any]:
	"""Compute weighted average scores across sentences by sentence length."""
	label_sums: dict[str, float] = {}
	total_weight = 0

	for result, weight in sentence_results:
		total_weight += weight
		for item in result:
			label = str(item["label"])
			score = float(item["score"])
			label_sums[label] = label_sums.get(label, 0.0) + (score * weight)

	averaged = {label: value / total_weight for label, value in label_sums.items()}
	top_label = max(averaged, key=averaged.get)

	return {
		"emotion": top_label.lower(),
		"confidence": float(averaged[top_label]),
		"all_scores": {label.lower(): round(score, 4) for label, score in averaged.items()},
	}


def detect_emotion(text: str) -> dict[str, Any]:
	"""Detect emotion from text and return aggregated emotion scores."""
	if not isinstance(text, str) or not text.strip():
		raise ValueError("text must be a non-empty string")

	pipe = _get_pipeline()
	normalized_text = text.strip()
	sentences = _split_sentences(normalized_text)
	if not sentences:
		sentences = [normalized_text]

	pairs: list[tuple[list[dict[str, Any]], int]] = []
	for sentence in sentences:
		# weight by sentence importance
		weight = len(sentence)
		pairs.append((pipe(sentence)[0], weight))
	return _aggregate(pairs)