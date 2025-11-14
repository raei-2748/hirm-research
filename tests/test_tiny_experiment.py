from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_tiny_experiment_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/run_experiment.py", "--config", "configs/experiments/tiny_test.yaml"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    output_dir = repo_root / "outputs" / "tiny_test"
    assert output_dir.exists()
    log_file = output_dir / "log.txt"
    assert log_file.exists()
    metrics_path = output_dir / "metrics.json"
    data = json.loads(metrics_path.read_text())
    assert "metrics" in data
