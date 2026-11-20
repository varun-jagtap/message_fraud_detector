import json
import subprocess
import sys
from pathlib import Path

import pytest

from message_fraud_detector.core import load_config, save_config
from message_fraud_detector.core import MessageFraudDetector
from message_fraud_detector.persistence import load_detector, save_detector
from message_fraud_detector.core import MutationGuardLite
from message_fraud_detector.sample import generate_records
from message_fraud_detector.core import TfIdfVectorizer
from message_fraud_detector.core import Example


def examples():
    return [
        Example("URGENT verify password and OTP now", "fraud", "test"),
        Example("Your account is suspended, send gift card", "fraud", "test"),
        Example("Please review the project agenda", "legitimate", "test"),
        Example("Dinner is confirmed for Tuesday", "legitimate", "test"),
    ]


def test_tfidf_fit_transform():
    vectorizer = TfIdfVectorizer().fit(["hello fraud", "hello bank"])
    vector = vectorizer.transform("hello fraud")
    assert vector
    assert vectorizer.cosine_sparse(vector, vector) == pytest.approx(1.0)


def test_mutationguard_detects_credentials():
    hits = MutationGuardLite().evaluate("URGENT! Verify your password and OTP now.")
    ids = {hit.rule_id for hit in hits}
    assert "credential_request" in ids
    assert "urgency_pressure" in ids


def test_mutationguard_detects_payment():
    hits = MutationGuardLite().evaluate("Send a gift card payment today")
    ids = {hit.rule_id for hit in hits}
    assert "payment_request" in ids


def test_detector_training_and_scoring():
    detector = MessageFraudDetector.train(examples())
    fraud = detector.score("URGENT verify password now")
    legitimate = detector.score("Review the project agenda")
    assert fraud.label in ("fraud", "suspicious")
    assert legitimate.label == "legitimate"
    assert fraud.reasons


def test_serialization_round_trip(tmp_path):
    detector = MessageFraudDetector.train(examples())
    path = tmp_path / "model.json"
    save_detector(detector, path)
    restored = load_detector(path)
    assert restored.score("verify password").to_dict()["label"]
    assert restored.index.size() == 4


def test_config_validation(tmp_path):
    config = load_config()
    path = tmp_path / "config.json"
    save_config(config, path)
    assert json.loads(path.read_text())["fraud_threshold"] == 0.48


def test_sample_generator_deterministic():
    first = generate_records(25, 99)
    second = generate_records(25, 99)
    assert first == second
    assert len(first) == 25


def test_batch_score():
    detector = MessageFraudDetector.train(examples())
    results = detector.score_many([
        ("m1", "URGENT password"),
        ("m2", "project meeting agenda"),
    ])
    assert len(results) == 2
    assert results[0].message_id == "m1"
