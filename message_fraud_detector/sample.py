"""Deterministic sample data generator for smoke tests and demos."""

import csv
import json
import random
from pathlib import Path
from typing import Iterable, Iterator, List


FRAUD_TEMPLATES = (
    "URGENT! Your bank account has been suspended. Verify your password and OTP immediately: https://secure-bank.example/reset",
    "Congratulations, you won a reward. Pay a $25 processing fee by gift card and send the code now.",
    "Police case detected. Transfer money by crypto today or legal action will follow.",
    "Amazon security alert: login and PIN required immediately. Call 1-800-555-0199.",
    "Your IRS refund is waiting. Send your SSN and bank login within 10 minutes.",
    "Limited offer: claim prize before it expires, send Bitcoin payment to verify.",
    "Your package will be deleted unless you verify account credentials at http://pay.example.",
    "Final warning! Your account is locked. Text password and OTP to +1 555 555 0112.",
)

LEGITIMATE_TEMPLATES = (
    "Your appointment is confirmed for Tuesday at 2 PM.",
    "The team meeting agenda is attached for your review.",
    "Thanks for your purchase. This message confirms our scheduled delivery.",
    "Please remember to submit the project status report by Friday.",
    "Dinner is at 7 PM at the restaurant near the station.",
    "The library loan is due next month. You can renew it online.",
    "Your monthly account statement is ready in the customer portal.",
    "Thanks for contacting support. We will reply during normal business hours.",
)

SUSPICIOUS_TEMPLATES = (
    "Please verify your account details when you have time.",
    "A payment request is pending; check the bank portal before proceeding.",
    "Limited-time offer available; review the reward terms.",
)


def generate_records(count: int = 100, seed: int = 7) -> List[dict]:
    if count < 1:
        raise ValueError("count must be at least 1")
    if count > 100000:
        raise ValueError("count must not exceed 100000")
    rng = random.Random(seed)
    records = []
    pools = (
        ("fraud", FRAUD_TEMPLATES),
        ("legitimate", LEGITIMATE_TEMPLATES),
        ("suspicious", SUSPICIOUS_TEMPLATES),
    )
    for index in range(count):
        label, pool = rng.choice(pools)
        text = rng.choice(pool)
        mutation = index % 5
        if mutation == 1:
            text = "Reminder: " + text
        elif mutation == 2:
            text = text + " Please act soon."
        elif mutation == 3:
            text = text.replace("account", "profile")
        records.append({
            "message_id": "sample-{:05d}".format(index + 1),
            "text": text,
            "label": label,
            "source": "sample_generator",
        })
    return records


def write_jsonl(records: Iterable[dict], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def write_csv(records: Iterable[dict], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    fieldnames = ["message_id", "text", "label", "source"]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})
