"""Lightweight message fraud detection package."""

from .core import (
    DetectorConfig,
    Example,
    MessageFraudDetector,
    RetrievalHit,
    RuleHit,
    ScoreResult,
)

__version__ = "1.0.0"

__all__ = [
    "DetectorConfig",
    "Example",
    "MessageFraudDetector",
    "RetrievalHit",
    "RuleHit",
    "ScoreResult",
]


def package_name() -> str:
    return "message_fraud_detector"
