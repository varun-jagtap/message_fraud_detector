"""Command-line interface for training, scoring, inspection, and samples."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

from .core import DetectorConfig, MessageFraudDetector, load_config, save_config
from .io import (
    read_records,
    records_to_examples,
    records_to_score_items,
    write_json,
    write_jsonl,
    ValidationError,
)
from .persistence import inspect_model, load_detector, save_detector
from .sample import generate_records, write_csv, write_jsonl as write_sample_jsonl


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frauddetector",
        description="TRE-style message fraud retrieval detector",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train and persist a detector")
    train.add_argument("--input", required=True, help="CSV or JSONL training file")
    train.add_argument("--format", choices=("csv", "jsonl", "json", "auto"), default="auto")
    train.add_argument("--config", help="Detector JSON configuration")
    train.add_argument("--model", required=True, help="Output model JSON")

    index = subparsers.add_parser("index", help="Alias for building an index")
    index.add_argument("--input", required=True, help="CSV or JSONL training file")
    index.add_argument("--format", choices=("csv", "jsonl", "json", "auto"), default="auto")
    index.add_argument("--config", help="Detector JSON configuration")
    index.add_argument("--model", required=True, help="Output model JSON")

    score = subparsers.add_parser("score", help="Score one message")
    score.add_argument("--model", required=True)
    score.add_argument("--text", help="Text to score")
    score.add_argument("--stdin", action="store_true", help="Read text from stdin")
    score.add_argument("--message-id", default="", help="Optional external message ID")
    score.add_argument(
        "--json",
        action="store_true",
        help="Print the complete machine-readable result as JSON",
    )

    batch = subparsers.add_parser("batch", help="Score CSV or JSONL records")
    batch.add_argument("--model", required=True)
    batch.add_argument("--input", help="Input file (omit or use - for stdin)")
    batch.add_argument("--format", choices=("csv", "jsonl", "json", "auto"), default="auto")
    batch.add_argument("--output", help="Output JSONL or JSON file")
    batch.add_argument("--output-format", choices=("jsonl", "json"), default="jsonl")

    inspect = subparsers.add_parser("inspect", help="Inspect a persisted model")
    inspect.add_argument("--model", required=True)

    generate = subparsers.add_parser("generate", help="Generate sample records")
    generate.add_argument("--count", type=positive_int, default=100)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--format", choices=("csv", "jsonl"), default="jsonl")
    generate.add_argument("--output", required=True)

    config_parser = subparsers.add_parser("config", help="Validate or create configuration")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("validate").add_argument("--file", required=True)
    create = config_sub.add_parser("create")
    create.add_argument("--output", required=True)
    return parser


def train_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    records = read_records(args.input, args.format)
    examples = records_to_examples(records)
    detector = MessageFraudDetector.train(examples, config)
    metadata = save_detector(detector, args.model)
    write_json(metadata, None, True)
    return 0


def score_command(args: argparse.Namespace) -> int:
    detector = load_detector(args.model)
    if args.stdin:
        text = sys.stdin.read()
    else:
        text = args.text
    if text is None:
        raise ValidationError("provide --text or --stdin")
    result = detector.score(text, args.message_id)
    if args.json:
        write_json(result.to_dict(), None, True)
    else:
        write_score_summary(result)
    return 0


def write_score_summary(result) -> None:
    """Print the compact human-facing score view."""
    print("Message fraud check")
    print("-------------------")
    print("Decision : {}".format(result.label.upper()))
    print("Score    : {:.1%}".format(result.score))
    print("Risk     : {}".format(result.risk_level.upper()))
    if result.message_id:
        print("Message  : {}".format(result.message_id))
    if result.reasons:
        print("Reasons  :")
        for reason in result.reasons[:3]:
            print("  - {}".format(reason))


def batch_command(args: argparse.Namespace) -> int:
    detector = load_detector(args.model)
    records = read_records(args.input, args.format)
    items = records_to_score_items(records)
    results = [detector.score(text, message_id).to_dict() for message_id, text in items]
    if args.output_format == "jsonl":
        write_jsonl(results, args.output)
    else:
        write_json(results, args.output, True)
    return 0


def inspect_command(args: argparse.Namespace) -> int:
    write_json(inspect_model(args.model), None, True)
    return 0


def generate_command(args: argparse.Namespace) -> int:
    records = generate_records(args.count, args.seed)
    if args.format == "csv":
        write_csv(records, args.output)
    else:
        write_sample_jsonl(records, args.output)
    write_json({"count": len(records), "path": args.output}, None, True)
    return 0


def config_command(args: argparse.Namespace) -> int:
    if args.config_command == "validate":
        config = load_config(args.file)
        write_json(config.to_dict(), None, True)
    else:
        save_config(DetectorConfig(), args.output)
        write_json({"path": args.output}, None, True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in ("train", "index"):
            return train_command(args)
        if args.command == "score":
            return score_command(args)
        if args.command == "batch":
            return batch_command(args)
        if args.command == "inspect":
            return inspect_command(args)
        if args.command == "generate":
            return generate_command(args)
        if args.command == "config":
            return config_command(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
