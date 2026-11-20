"""Batch reporting and JSON-safe result formatting."""

from collections import Counter
from typing import Iterable, Mapping, Sequence

from .core import ScoreResult


def summarize_results(results: Sequence[ScoreResult]) -> dict:
    if not results:
        return {
            "count": 0,
            "labels": {},
            "risk_levels": {},
            "average_score": 0.0,
            "maximum_score": 0.0,
        }
    labels = Counter(result.label for result in results)
    risks = Counter(result.risk_level for result in results)
    scores = [result.score for result in results]
    return {
        "count": len(results),
        "labels": dict(sorted(labels.items())),
        "risk_levels": dict(sorted(risks.items())),
        "average_score": round(sum(scores) / len(scores), 6),
        "maximum_score": round(max(scores), 6),
        "minimum_score": round(min(scores), 6),
    }


def filter_by_label(results: Iterable[ScoreResult], label: str) -> list[ScoreResult]:
    normalized = label.lower().strip()
    if normalized not in ("fraud", "legitimate", "suspicious"):
        raise ValueError("unsupported result label: {}".format(label))
    return [result for result in results if result.label == normalized]


def filter_by_minimum_score(
    results: Iterable[ScoreResult],
    minimum_score: float,
) -> list[ScoreResult]:
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between 0 and 1")
    return [result for result in results if result.score >= minimum_score]


def rank_results(results: Iterable[ScoreResult], limit: int = 10) -> list[ScoreResult]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    return sorted(results, key=lambda result: (-result.score, result.message_id))[:limit]


def to_jsonl_records(results: Iterable[ScoreResult]) -> list[dict]:
    return [result.to_dict() for result in results]
