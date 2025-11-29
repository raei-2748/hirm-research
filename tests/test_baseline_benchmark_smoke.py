import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")


def _subprocess_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd()}:{env.get('PYTHONPATH', '')}"
    return env


def _write_minimal_config(tmp_path: Path) -> Path:
    yaml_text = """
defaults:
  - env: synthetic_heston
  - model: mlp_small
  - objective: erm
  - _self_
experiment:
  name: baseline_smoke
training:
  max_epochs: 2
  batch_size: 8
  early_stop_patience: 1
diagnostics:
  enabled: true
  crisis:
    enabled: false
datasets: [synthetic_heston]
methods: [erm]
seeds: [0]
"""
    path = tmp_path / "config_smoke.yaml"
    path.write_text(yaml_text)
    return path


@pytest.mark.smoke
def test_run_grid_and_results(tmp_path):
    cfg_path = _write_minimal_config(tmp_path)
    results_root = tmp_path / "baseline_results"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_grid.py",
            "--config",
            str(cfg_path),
            "--mode",
            "benchmark",
            "--datasets",
            "synthetic_heston",
            "--methods",
            "erm,hirm",
            "--seeds",
            "0",
            "--device",
            "cpu",
            "--results-dir",
            str(results_root),
        ],
        env=_subprocess_env(),
    )

    for method in ("erm", "hirm"):
        base = results_root / "synthetic_heston" / method / "seed_0"
        assert base.exists()
        for fname in ("train_logs.jsonl", "checkpoint.pt", "diagnostics.jsonl", "config.json", "metadata.json"):
            assert (base / fname).exists(), fname


@pytest.mark.smoke
def test_integrity_checker(tmp_path):
    cfg_path = _write_minimal_config(tmp_path)
    results_dir = tmp_path / "results"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_grid.py",
            "--config",
            str(cfg_path),
            "--mode",
            "benchmark",
            "--datasets",
            "synthetic_heston",
            "--methods",
            "erm",
            "--seeds",
            "0",
            "--device",
            "cpu",
            "--results-dir",
            str(results_dir),
        ],
        env=_subprocess_env(),
    )
    try:
        subprocess.check_call(
            [sys.executable, "scripts/check_baseline_integrity.py", "--config", str(cfg_path), "--results-root", str(results_dir)],
            env=_subprocess_env(),
        )
    except subprocess.CalledProcessError:
        print(f"DEBUG: Integrity check failed. Results dir contents: {list(results_dir.rglob('*'))}", file=sys.stderr)
        raise


@pytest.mark.smoke
def test_summarizer(tmp_path):
    cfg_path = _write_minimal_config(tmp_path)
    results_dir = tmp_path / "results"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_grid.py",
            "--config",
            str(cfg_path),
            "--mode",
            "benchmark",
            "--datasets",
            "synthetic_heston",
            "--methods",
            "erm",
            "--seeds",
            "0",
            "--device",
            "cpu",
            "--results-dir",
            str(results_dir),
        ],
        env=_subprocess_env(),
    )
    subprocess.check_call(
        [sys.executable, "scripts/summarize_baseline_results.py", "--config", str(cfg_path), "--results-root", str(results_dir)],
        env=_subprocess_env(),
    )
    summary_csv = results_dir / "summary" / "synthetic_heston_summary.csv"
    summary_json = results_dir / "summary" / "synthetic_heston_summary.json"
    
    # Debug info on failure
    if not summary_csv.exists():
        print(f"DEBUG: Results dir contents: {list(results_dir.rglob('*'))}", file=sys.stderr)
        
    assert summary_csv.exists(), f"Summary CSV missing. Contents: {list(results_dir.rglob('*'))}"
    assert summary_json.exists(), "Summary JSON missing"
    assert summary_csv.stat().st_size > 0, f"Summary CSV is empty. Content: {summary_csv.read_text() if summary_csv.exists() else 'N/A'}"
    assert summary_json.stat().st_size > 0, "Summary JSON is empty"
