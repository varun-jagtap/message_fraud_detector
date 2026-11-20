import subprocess
import sys
from pathlib import Path


def test_benchmark_script_runs(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    csv_output = tmp_path / "benchmark.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_methods.py",
            "--count",
            "60",
            "--seed",
            "11",
            "--test-fraction",
            "0.30",
            "--csv",
            str(csv_output),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "TRE-style retrieval" in result.stdout
    assert "MutationGuard-lite" in result.stdout
    assert "Combined 55/45" in result.stdout
    assert "Accuracy" in result.stdout
    assert "Msgs/sec" in result.stdout
    assert csv_output.exists()
    assert "method,count,accuracy" in csv_output.read_text()
