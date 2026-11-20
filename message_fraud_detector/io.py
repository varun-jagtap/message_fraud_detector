from .core import Example

"""Input validation helpers and clear validation errors."""

import csv
from typing import Iterable, Iterator, Mapping


class ValidationError(ValueError):
    pass


def validate_text(text: object, line_number: int = 1) -> str:
    if not isinstance(text, str):
        raise ValidationError("text at line {} must be a string".format(line_number))
    text = text.strip()
    if not text:
        raise ValidationError("text at line {} must not be empty".format(line_number))
    if len(text) > 100000:
        raise ValidationError("text at line {} exceeds 100000 characters".format(line_number))
    return text


def validate_label(label: object, line_number: int = 1) -> str:
    if not isinstance(label, str):
        raise ValidationError("label at line {} must be a string".format(line_number))
    label = label.strip().lower()
    allowed = {"fraud", "legitimate", "suspicious", "unknown"}
    if label not in allowed:
        raise ValidationError(
            "label at line {} must be one of {}".format(
                line_number,
                ", ".join(sorted(allowed)),
            )
        )
    return label


def validate_message_id(message_id: object) -> str:
    if message_id is None:
        return ""
    if not isinstance(message_id, str):
        raise ValidationError("message_id must be a string")
    if len(message_id) > 200:
        raise ValidationError("message_id must not exceed 200 characters")
    return message_id

"""Format conversion helpers for model and batch interoperability."""

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def csv_to_jsonl(source: str, destination: str) -> int:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.exists():
        raise FileNotFoundError(source)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source_path.open("r", encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        if "text" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain a text column")
        with destination_path.open("w", encoding="utf-8") as destination_handle:
            for row in reader:
                destination_handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True)
                )
                destination_handle.write("\n")
                count += 1
    return count


def jsonl_to_csv(
    source: str,
    destination: str,
    fieldnames: Sequence[str] = ("message_id", "text", "label", "source"),
) -> int:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.exists():
        raise FileNotFoundError(source)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source_path.open("r", encoding="utf-8") as source_handle:
        with destination_path.open("w", encoding="utf-8-sig", newline="") as destination_handle:
            writer = csv.DictWriter(destination_handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for line_number, line in enumerate(source_handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("invalid JSONL at line {}".format(line_number)) from exc
                if not isinstance(record, dict):
                    raise ValueError("JSONL line must be an object")
                writer.writerow({
                    field: record.get(field, "")
                    for field in fieldnames
                })
                count += 1
    return count


def compact_jsonl(records: Iterable[Mapping[str, object]], destination: str) -> int:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count

"""CSV, JSONL, stdin, and file input/output utilities."""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, TextIO



def read_jsonl(path: str | os.PathLike[str] | None = None) -> Iterator[dict]:
    handle = sys.stdin if path in (None, "-", "") else open(path, "r", encoding="utf-8")
    try:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(
                    "invalid JSONL at line {}: {}".format(line_number, exc.msg)
                ) from exc
            if not isinstance(value, dict):
                raise ValidationError("JSONL line {} must be an object".format(line_number))
            yield value
    finally:
        if handle is not sys.stdin:
            handle.close()


def read_csv(path: str | os.PathLike[str] | None = None) -> Iterator[dict]:
    handle = sys.stdin if path in (None, "-", "") else open(path, "r", encoding="utf-8-sig", newline="")
    try:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError("CSV input has no header")
        if "text" not in reader.fieldnames:
            raise ValidationError("CSV input must contain a 'text' column")
        for row_number, row in enumerate(reader, start=2):
            yield {"__csv_line": row_number, **row}
    finally:
        if handle is not sys.stdin:
            handle.close()


def read_records(
    path: str | os.PathLike[str] | None,
    input_format: str,
) -> Iterator[dict]:
    if input_format not in ("csv", "jsonl", "json", "auto"):
        raise ValidationError("unsupported input format: {}".format(input_format))
    if input_format == "auto":
        if path in (None, "-", ""):
            input_format = "jsonl"
        else:
            input_format = Path(path).suffix.lower().lstrip(".")
            if input_format not in ("csv", "jsonl", "json"):
                raise ValidationError("cannot infer input format from file extension")
    if input_format == "csv":
        yield from read_csv(path)
    elif input_format == "jsonl":
        yield from read_jsonl(path)
    elif input_format == "json":
        handle = sys.stdin if path in (None, "-", "") else open(path, "r", encoding="utf-8")
        try:
            value = json.load(handle)
        finally:
            if handle is not sys.stdin:
                handle.close()
        if not isinstance(value, list):
            raise ValidationError("JSON input must contain a list")
        for index, record in enumerate(value, start=1):
            if not isinstance(record, dict):
                raise ValidationError("JSON record {} must be an object".format(index))
            yield record


def records_to_examples(records: Iterable[dict]) -> List[Example]:
    examples: List[Example] = []
    for index, record in enumerate(records, start=1):
        line_number = record.pop("__csv_line", index)
        text = validate_text(record.get("text"), line_number)
        label = validate_label(record.get("label", "unknown"), line_number)
        message_id = validate_message_id(record.get("message_id"))
        source = record.get("source", "input")
        if not isinstance(source, str):
            source = str(source)
        excluded = {"text", "label", "message_id", "source"}
        metadata = {key: value for key, value in record.items() if key not in excluded}
        examples.append(Example(
            text=text,
            label=label,
            source=source,
            message_id=message_id,
            metadata=metadata,
        ))
    return examples


def records_to_score_items(records: Iterable[dict]) -> List[tuple[str, str]]:
    items = []
    for index, record in enumerate(records, start=1):
        line_number = record.pop("__csv_line", index)
        text = validate_text(record.get("text"), line_number)
        message_id = validate_message_id(record.get("message_id"))
        items.append((message_id, text))
    return items


def write_json(value: object, path: str | os.PathLike[str] | None = None, pretty: bool = True) -> None:
    if pretty:
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if path in (None, "-", ""):
        sys.stdout.write(payload)
        sys.stdout.write("\n")
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")
    os.replace(temporary, target)


def write_jsonl(records: Iterable[object], path: str | os.PathLike[str] | None = None) -> None:
    handle = sys.stdout if path in (None, "-", "") else open(path, "w", encoding="utf-8")
    try:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    finally:
        if handle is not sys.stdout:
            handle.close()

"""Schema definitions and structural validation for public JSON records."""

from typing import Iterable, Mapping, Sequence


TRAINING_FIELDS = {
    "text",
    "label",
    "message_id",
    "source",
}

SCORING_FIELDS = {
    "text",
    "message_id",
    "source",
}


def require_mapping(value: object, name: str = "record") -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a JSON object".format(name))
    return value


def validate_training_record(record: Mapping, index: int = 1) -> None:
    require_mapping(record, "training record")
    if "text" not in record:
        raise ValueError("training record {} is missing text".format(index))
    if not isinstance(record["text"], str):
        raise ValueError("training record {} text must be a string".format(index))
    if not record["text"].strip():
        raise ValueError("training record {} text must not be empty".format(index))
    if "label" in record:
        if not isinstance(record["label"], str):
            raise ValueError("training record {} label must be a string".format(index))
        allowed = {"fraud", "legitimate", "suspicious", "unknown"}
        if record["label"].lower().strip() not in allowed:
            raise ValueError("training record {} has unsupported label".format(index))


def validate_scoring_record(record: Mapping, index: int = 1) -> None:
    require_mapping(record, "scoring record")
    if "text" not in record:
        raise ValueError("scoring record {} is missing text".format(index))
    if not isinstance(record["text"], str):
        raise ValueError("scoring record {} text must be a string".format(index))
    if not record["text"].strip():
        raise ValueError("scoring record {} text must not be empty".format(index))


def validate_training_records(records: Iterable[Mapping]) -> int:
    count = 0
    for index, record in enumerate(records, start=1):
        validate_training_record(record, index)
        count += 1
    if count == 0:
        raise ValueError("training records must not be empty")
    return count


def validate_scoring_records(records: Iterable[Mapping]) -> int:
    count = 0
    for index, record in enumerate(records, start=1):
        validate_scoring_record(record, index)
        count += 1
    return count


def result_schema() -> dict:
    return {
        "type": "object",
        "required": [
            "message_id",
            "text",
            "label",
            "score",
            "confidence",
            "risk_level",
            "rule_hits",
            "retrieval_hits",
            "reasons",
        ],
        "properties": {
            "message_id": {"type": "string"},
            "text": {"type": "string"},
            "label": {"enum": ["fraud", "legitimate", "suspicious"]},
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "risk_level": {"enum": ["low", "medium", "high"]},
            "rule_hits": {"type": "array"},
            "retrieval_hits": {"type": "array"},
            "reasons": {"type": "array", "items": {"type": "string"}},
        },
    }

"""Safe text and metadata sanitization for stored examples and results."""

import re
import unicodedata
from typing import Iterable, Mapping


CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)
SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"\b(?:password|passcode|otp|pin|secret|token)\s*[:=]\s*\S+",
    re.I,
)


def remove_control_characters(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return CONTROL_PATTERN.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_unicode(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return unicodedata.normalize("NFKC", text)


def sanitize_text(text: str, max_length: int = 100000) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if max_length < 1:
        raise ValueError("max_length must be positive")
    sanitized = normalize_unicode(text)
    sanitized = remove_control_characters(sanitized)
    sanitized = collapse_whitespace(sanitized)
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized


def redact_secrets(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return SECRET_ASSIGNMENT_PATTERN.sub("[REDACTED]", text)


def sanitize_metadata(metadata: Mapping) -> dict:
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    clean = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        if len(key) > 200:
            raise ValueError("metadata key exceeds 200 characters")
        if isinstance(value, str):
            clean[key] = sanitize_text(value, 10000)
        elif isinstance(value, (int, float, bool)) or value is None:
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = [
                sanitize_text(str(item), 10000)
                if not isinstance(item, (dict, list))
                else str(item)[:10000]
                for item in value
            ]
        else:
            clean[key] = str(value)[:10000]
    return clean
