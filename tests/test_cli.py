import json
import subprocess
import sys
from pathlib import Path


def run_cli(repo, *args):
    return subprocess.run(
        [sys.executable, "-m", "message_fraud_detector", *args],
        cwd=repo,
        text=True,
        capture_output=True,
    )


def test_generate_and_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    sample = tmp_path / "sample.jsonl"
    model = tmp_path / "model.json"
    generated = run_cli(
        repo,
        "generate",
        "--count",
        "30",
        "--seed",
        "3",
        "--format",
        "jsonl",
        "--output",
        str(sample),
    )
    assert generated.returncode == 0
    trained = run_cli(
        repo,
        "train",
        "--input",
        str(sample),
        "--format",
        "jsonl",
        "--model",
        str(model),
    )
    assert trained.returncode == 0
    scored = run_cli(
        repo,
        "score",
        "--model",
        str(model),
        "--text",
        "URGENT verify password now",
        "--json",
    )
    assert scored.returncode == 0
    payload = json.loads(scored.stdout)
    assert payload["label"] in ("fraud", "suspicious", "legitimate")


def test_score_defaults_to_compact_human_output(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    sample = tmp_path / "sample.jsonl"
    model = tmp_path / "model.json"
    assert run_cli(repo, "generate", "--count", "20", "--output", str(sample)).returncode == 0
    assert run_cli(repo, "train", "--input", str(sample), "--model", str(model)).returncode == 0
    scored = run_cli(
        repo,
        "score",
        "--model",
        str(model),
        "--text",
        "URGENT verify password now",
    )
    assert scored.returncode == 0
    assert "Message fraud check" in scored.stdout
    assert "Decision" in scored.stdout
    assert "Score" in scored.stdout
    assert "retrieval_hits" not in scored.stdout


def test_batch_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    sample = tmp_path / "sample.jsonl"
    model = tmp_path / "model.json"
    assert run_cli(repo, "generate", "--count", "20", "--output", str(sample)).returncode == 0
    assert run_cli(repo, "train", "--input", str(sample), "--model", str(model)).returncode == 0
    output = tmp_path / "scores.jsonl"
    result = run_cli(
        repo,
        "batch",
        "--model",
        str(model),
        "--input",
        str(sample),
        "--output",
        str(output),
        "--output-format",
        "jsonl",
    )
    assert result.returncode == 0
    assert len(output.read_text().strip().splitlines()) == 20
