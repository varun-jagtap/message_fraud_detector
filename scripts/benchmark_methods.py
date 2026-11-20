#!/usr/bin/env python3
"""Compare TRE-style retrieval, MutationGuard-lite, and combined detection."""

import argparse
import csv
import random
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from message_fraud_detector.core import (
    DetectorConfig,
    Example,
    MessageFraudDetector,
    MutationGuardLite,
    RetrievalIndex,
    TfIdfVectorizer,
)
from message_fraud_detector.io import read_records, records_to_examples
from message_fraud_detector.sample import generate_records


def stratified_split(
    examples: Sequence[Example],
    test_fraction: float,
    seed: int,
) -> tuple[List[Example], List[Example]]:
    """Split each label group without mixing identical label distributions."""
    rng = random.Random(seed)
    train: List[Example] = []
    test: List[Example] = []
    for label in sorted({example.label for example in examples}):
        group = [example for example in examples if example.label == label]
        rng.shuffle(group)
        test_count = max(1, int(round(len(group) * test_fraction)))
        test.extend(group[:test_count])
        train.extend(group[test_count:])
    if not train:
        raise ValueError("training split is empty; add more examples")
    if not test:
        raise ValueError("test split is empty")
    return train, test


def positive(label: str) -> bool:
    """Treat fraud and suspicious as positive fraud-risk outcomes."""
    return label.lower() in ("fraud", "suspicious")


def metrics(actual: Sequence[str], predicted: Sequence[str]) -> dict[str, float | int]:
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for wanted, got in zip(actual, predicted):
        wanted_positive = positive(wanted)
        got_positive = positive(got)
        if wanted_positive and got_positive:
            true_positive += 1
        elif not wanted_positive and not got_positive:
            true_negative += 1
        elif got_positive:
            false_positive += 1
        else:
            false_negative += 1

    total = len(actual)
    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


def table_value(value: object, width: int) -> str:
    text = str(value)
    return text[:width].ljust(width)


def print_table(rows: Sequence[dict[str, object]]) -> None:
    columns = (
        ("Method", "method", 24),
        ("Accuracy", "accuracy", 10),
        ("Precision", "precision", 10),
        ("Recall", "recall", 10),
        ("F1", "f1", 10),
        ("Avg ms", "avg_ms", 10),
        ("Msgs/sec", "throughput", 12),
        ("Peak MB", "peak_mb", 9),
        ("FP", "fp", 5),
        ("FN", "fn", 5),
    )
    header = "| " + " | ".join(table_value(name, width) for name, key, width in columns) + " |"
    separator = "|" + "|".join("-" * (width + 2) for name, key, width in columns) + "|"
    print(header)
    print(separator)
    for row in rows:
        values = []
        for name, key, width in columns:
            value = row[key]
            if key in ("accuracy", "precision", "recall", "f1"):
                value = "{:.2%}".format(float(value))
            elif key in ("avg_ms",):
                value = "{:.3f}".format(float(value))
            elif key in ("throughput",):
                value = "{:.1f}".format(float(value))
            elif key in ("peak_mb",):
                value = "{:.2f}".format(float(value))
            values.append(table_value(value, width))
        print("| " + " | ".join(values) + " |")


def timed_score(call: callable, texts: Sequence[str]) -> tuple[List[str], float, float]:
    tracemalloc.start()
    predicted: List[str] = []
    started = time.perf_counter()
    for text in texts:
        predicted.append(call(text))
    elapsed = time.perf_counter() - started
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return predicted, elapsed, peak / (1024 * 1024)


def train_retrieval(
    examples: Sequence[Example],
    config: DetectorConfig,
) -> RetrievalIndex:
    vectorizer = TfIdfVectorizer(
        max_features=config.max_features,
        ngram_min=config.ngram_min,
        ngram_max=config.ngram_max,
        sublinear_tf=config.sublinear_tf,
        strip_accents=config.strip_accents,
    )
    return RetrievalIndex(vectorizer).fit(examples)


def retrieval_prediction(
    index: RetrievalIndex,
    config: DetectorConfig,
    text: str,
) -> str:
    label, score, hits = index.nearest_label(text, config.top_k)
    if score < config.retrieval_min_score:
        return "legitimate"
    if label in ("fraud", "suspicious"):
        if label == "fraud" and score >= config.suspicious_threshold:
            return "fraud"
        if label == "suspicious":
            return "suspicious"
    return "legitimate"


def rule_prediction(guard: MutationGuardLite, config: DetectorConfig, text: str) -> str:
    hits = guard.evaluate(text)
    score = guard.weighted_score(hits, config.rule_cap)
    if score >= config.fraud_threshold:
        return "fraud"
    if score >= config.suspicious_threshold:
        return "suspicious"
    return "legitimate"


def benchmark(
    train_examples: Sequence[Example],
    test_examples: Sequence[Example],
) -> List[dict[str, object]]:
    config = DetectorConfig()
    config.validate()
    actual = [example.label for example in test_examples]
    texts = [example.text for example in test_examples]

    started = time.perf_counter()
    index = train_retrieval(train_examples, config)
    retrieval_train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    combined = MessageFraudDetector.train(train_examples, config)
    combined_train_seconds = time.perf_counter() - started
    guard = MutationGuardLite()

    methods = (
        (
            "TRE-style retrieval",
            lambda text: retrieval_prediction(index, config, text),
        ),
        (
            "MutationGuard-lite",
            lambda text: rule_prediction(guard, config, text),
        ),
        (
            "Combined 55/45",
            lambda text: combined.score(text).label,
        ),
    )

    rows: List[dict[str, object]] = []
    for name, prediction in methods:
        predicted, elapsed, peak_mb = timed_score(prediction, texts)
        result = metrics(actual, predicted)
        if name == "TRE-style retrieval":
            train_seconds = retrieval_train_seconds
        elif name == "MutationGuard-lite":
            train_seconds = 0.0
        else:
            train_seconds = combined_train_seconds
        result.update({
            "method": name,
            "train_ms": train_seconds * 1000,
            "avg_ms": elapsed * 1000 / len(texts),
            "throughput": len(texts) / elapsed if elapsed else 0.0,
            "peak_mb": peak_mb,
        })
        rows.append(result)
    return rows


def load_examples(args: argparse.Namespace) -> List[Example]:
    if args.input:
        records = read_records(args.input, args.format)
        examples = records_to_examples(records)
    else:
        records = generate_records(args.count, args.seed)
        examples = records_to_examples(records)
    if len(examples) < 4:
        raise ValueError("benchmark needs at least 4 labeled examples")
    labels = {example.label for example in examples}
    if "legitimate" not in labels:
        raise ValueError("benchmark needs at least one legitimate example")
    if not labels.intersection({"fraud", "suspicious"}):
        raise ValueError("benchmark needs at least one fraud-risk example")
    return examples


def write_csv(path: str, rows: Sequence[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "method", "count", "accuracy", "precision", "recall", "f1",
        "train_ms", "avg_ms", "throughput", "peak_mb", "tp", "tn", "fp", "fn",
    )
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval, behavioral, and combined fraud detection.",
    )
    parser.add_argument("--input", help="CSV or JSONL benchmark dataset")
    parser.add_argument(
        "--format",
        choices=("csv", "jsonl", "json", "auto"),
        default="auto",
    )
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--csv", help="Optional CSV output path")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.05 <= args.test_fraction < 0.95:
        parser.error("--test-fraction must be between 0.05 and 0.95")
    if args.count < 4:
        parser.error("--count must be at least 4")
    try:
        examples = load_examples(args)
        train_examples, test_examples = stratified_split(
            examples,
            args.test_fraction,
            args.seed,
        )
        rows = benchmark(train_examples, test_examples)
        print("Fraud benchmark: fraud/suspicious are positive")
        print("Train: {} examples | Test: {} examples".format(
            len(train_examples),
            len(test_examples),
        ))
        print_table(rows)
        if args.csv:
            write_csv(args.csv, rows)
            print("CSV: {}".format(args.csv))
        return 0
    except (ValueError, OSError, csv.Error) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
