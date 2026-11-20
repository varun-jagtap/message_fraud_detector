import json
from pathlib import Path

import pytest

from message_fraud_detector.io import read_records, records_to_examples, write_json
from message_fraud_detector.io import ValidationError


def test_jsonl_reader(tmp_path):
    path = tmp_path / "training.jsonl"
    path.write_text(
        '{"text":"hello","label":"legitimate","message_id":"1"}\n'
        '\n'
        '{"text":"urgent password","label":"fraud","message_id":"2"}\n',
        encoding="utf-8",
    )
    records = list(read_records(path, "jsonl"))
    examples = records_to_examples(records)
    assert len(examples) == 2
    assert examples[1].label == "fraud"


def test_csv_reader(tmp_path):
    path = tmp_path / "training.csv"
    path.write_text(
        "message_id,text,label\n"
        "a,hello world,legitimate\n",
        encoding="utf-8",
    )
    records = list(read_records(path, "csv"))
    assert records[0]["text"] == "hello world"


def test_json_reader(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(json.dumps([{"text": "hello", "label": "legitimate"}]))
    records = list(read_records(path, "json"))
    assert len(records) == 1


def test_empty_text_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": " ", "label": "fraud"}))
    with pytest.raises(ValidationError):
        records_to_examples(read_records(path, "jsonl"))


def test_write_json(tmp_path):
    path = tmp_path / "nested" / "output.json"
    write_json({"ok": True}, path)
    assert json.loads(path.read_text())["ok"] is True
