"""Offline evaluation and simple holdout metrics."""

from typing import Iterable, List, Sequence

from .core import MessageFraudDetector
from .core import Example, ScoreResult


def confusion_matrix(expected: Sequence[str], predicted: Sequence[str], labels: Sequence[str]) -> dict:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths must match")
    matrix = {label: {other: 0 for other in labels} for label in labels}
    for actual, guess in zip(expected, predicted):
        if actual not in matrix:
            raise ValueError("unknown actual label: {}".format(actual))
        if guess not in matrix[actual]:
            raise ValueError("unknown predicted label: {}".format(guess))
        matrix[actual][guess] += 1
    return matrix


def binary_metrics(expected: Sequence[str], predicted: Sequence[str]) -> dict:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths must match")
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for actual, guess in zip(expected, predicted):
        actual_fraud = actual.lower() == "fraud"
        predicted_fraud = guess.lower() == "fraud"
        if actual_fraud and predicted_fraud:
            true_positive += 1
        elif not actual_fraud and predicted_fraud:
            false_positive += 1
        elif not actual_fraud and not predicted_fraud:
            true_negative += 1
        else:
            false_negative += 1
    precision = divide(true_positive, true_positive + false_positive)
    recall = divide(true_positive, true_positive + false_negative)
    specificity = divide(true_negative, true_negative + false_positive)
    accuracy = divide(true_positive + true_negative, len(expected))
    f1 = divide(2 * precision * recall, precision + recall)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1": round(f1, 6),
    }


def divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def evaluate_detector(
    detector: MessageFraudDetector,
    examples: Sequence[Example],
) -> dict:
    if not examples:
        raise ValueError("evaluation requires examples")
    actual = []
    predicted = []
    detailed = []
    for example in examples:
        result = detector.score(example.text, example.message_id)
        actual.append(example.label.lower())
        predicted.append(result.label)
        detailed.append({
            "message_id": example.message_id,
            "expected": example.label.lower(),
            "predicted": result.label,
            "score": result.score,
        })
    labels = sorted(set(actual) | set(predicted))
    return {
        "count": len(examples),
        "binary": binary_metrics(actual, predicted),
        "confusion_matrix": confusion_matrix(actual, predicted, labels),
        "details": detailed,
    }


def leave_one_out_accuracy(
    examples: Sequence[Example],
    detector_factory,
) -> float:
    if len(examples) < 2:
        raise ValueError("leave-one-out needs at least two examples")
    correct = 0
    for held_index in range(len(examples)):
        training = [
            example
            for index, example in enumerate(examples)
            if index != held_index
        ]
        detector = detector_factory(training)
        held = examples[held_index]
        result = detector.score(held.text, held.message_id)
        if result.label == held.label.lower():
            correct += 1
    return correct / len(examples)
