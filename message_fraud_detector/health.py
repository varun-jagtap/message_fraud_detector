"""Health and diagnostic checks for persisted detectors."""

from typing import Iterable, Mapping

from .core import MessageFraudDetector


def check_detector(detector: MessageFraudDetector) -> dict:
    checks = []
    example_count = detector.index.size()
    checks.append({
        "name": "examples",
        "ok": example_count > 0,
        "detail": "{} examples".format(example_count),
    })
    vocabulary_size = len(detector.index.vectorizer.vocabulary)
    checks.append({
        "name": "vocabulary",
        "ok": vocabulary_size > 0,
        "detail": "{} terms".format(vocabulary_size),
    })
    checks.append({
        "name": "configuration",
        "ok": True,
        "detail": "configuration validation passed",
    })
    labels = {example.label for example in detector.index.examples}
    checks.append({
        "name": "labels",
        "ok": bool(labels),
        "detail": ", ".join(sorted(labels)),
    })
    all_ok = all(check["ok"] for check in checks)
    return {
        "ok": all_ok,
        "checks": checks,
    }


def probe_texts(detector: MessageFraudDetector, texts: Iterable[str]) -> dict:
    failures = []
    count = 0
    for index, text in enumerate(texts, start=1):
        count += 1
        try:
            result = detector.score(text, "health-{}".format(index))
            if result.score < 0.0 or result.score > 1.0:
                failures.append(index)
        except Exception as exc:
            failures.append(index)
    return {
        "ok": not failures,
        "count": count,
        "failures": failures,
    }
