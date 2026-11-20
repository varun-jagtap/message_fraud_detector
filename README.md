# Message Fraud Detector

An isolated, standard-library-only Python CLI for detecting potentially fraudulent
messages. It combines TRE-style TF-IDF nearest-neighbor retrieval with
MutationGuard-lite behavioral rules. It does not download models or require a
heavy ML framework.

## Features

- Pure-Python TF-IDF with unigram/bigram terms and cosine similarity
- Weighted, explainable retrieval nearest-neighbor index
- Urgency, credential, payment, threat, impersonation, link, and stylistic rules
- CSV, JSONL, and JSON input/output
- Atomic JSON model and configuration persistence
- Training, indexing, scoring, batch scoring, inspection, and sample generation
- Deterministic sample corpus, validation, diagnostics, and unit tests
- 20+ realistic templates each for legitimate, suspicious, and fraud messages

## Install

```bash
cd message_fraud_detector
python -m pip install -e .
```

## Quick start

Generate a sample:

```bash
python -m message_fraud_detector generate --count 100 --format jsonl --output sample.jsonl
```

Train/index:

```bash
python -m message_fraud_detector train --input sample.jsonl --format jsonl --model model.json
# The index command is an alias:
python -m message_fraud_detector index --input sample.jsonl --format jsonl --model model.json
```

Score text:

```bash
python -m message_fraud_detector score --model model.json --text "URGENT: verify password and OTP now"
printf "Your appointment is confirmed for Tuesday." | python -m message_fraud_detector score --model model.json --stdin
```

The default score output is a compact human-readable summary containing the
decision, score, risk level, and up to three reasons. Use `--json` when the
complete result is needed for scripts or debugging:

```bash
python -m message_fraud_detector score --model model.json --text "URGENT: verify password and OTP now" --json
```

Batch scoring:

```bash
python -m message_fraud_detector batch --model model.json --input sample.jsonl --format jsonl --output scores.jsonl --output-format jsonl
```

Inspect:

```bash
python -m message_fraud_detector inspect --model model.json
```

## Input formats

Training CSV requires a `text` column and accepts `label`, `message_id`, and
`source`. JSONL and JSON records use the same fields. Labels are `fraud`,
`legitimate`, `suspicious`, or `unknown`. Batch scoring only requires `text`.

Example JSONL:

```json
{"message_id":"m1","text":"URGENT verify account password now","label":"fraud","source":"review"}
{"message_id":"m2","text":"Your delivery is scheduled for Monday","label":"legitimate","source":"review"}
```

## Configuration

Create a default configuration:

```bash
python -m message_fraud_detector config create --output detector.json
python -m message_fraud_detector config validate --file detector.json
```

The JSON fields include `retrieval_weight`, `rule_weight`,
`fraud_threshold`, `suspicious_threshold`, `top_k`,
`retrieval_min_score`, `rule_cap`, `max_features`, `ngram_min`,
`ngram_max`, `sublinear_tf`, `strip_accents`, and `model_version`.
Environment variables use the `MESSAGE_FRAUD_` prefix, for example
`MESSAGE_FRAUD_FRAUD_THRESHOLD`.

## Scoring model

Final score = retrieval weight × nearest retrieved label score +
rule weight × capped rule score. Labels are:

- `fraud`: score at or above the fraud threshold
- `suspicious`: score at or above the suspicious threshold
- `legitimate`: otherwise

Retrieval and rule evidence are included in JSON output.

## Tests

```bash
python -m pytest -q
```

## Method benchmark

Run a side-by-side benchmark using generated labeled messages:

```bash
python scripts/benchmark_methods.py --count 300 --seed 7 --test-fraction 0.30
```

Use your own CSV or JSONL dataset:

```bash
python scripts/benchmark_methods.py --input data.jsonl --format jsonl --csv benchmark.csv
```

The benchmark compares TRE-style retrieval, MutationGuard-lite rules, and the
combined 55/45 detector. It reports accuracy, precision, recall, F1, average
scoring latency, throughput, peak benchmark memory, false positives, and false
negatives. Fraud and suspicious labels are both treated as positive outcomes.

## Web interface

Start the local browser interface:

```bash
python webapp/server.py
```

Open `http://127.0.0.1:8080/`.

The web interface exposes single-message scoring, JSON/CSV/JSONL batch
scoring, model training, sample generation, model inspection, and configuration
creation and validation.

Saved models are discovered automatically throughout the project directory.
New models are saved under `web_data/`; existing model JSON files in the project
root or subdirectories are also shown in the web model selectors.
