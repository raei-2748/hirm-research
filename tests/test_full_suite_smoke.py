import subprocess
import sys
from pathlib import Path


def test_full_suite_smoke(tmp_path):
    results_root = tmp_path / "full_suite"

    cmd = [
        sys.executable,
        "scripts/run_full_experiment_suite.py",
        "--config",
        "configs/experiments/full_experiment_suite.yaml",
        "--datasets",
        "synthetic_heston,real_spy",
        "--methods",
        "erm_baseline,hirm_full",
        "--seeds",
        "0",
        "--device",
        "cpu",
        "--reduced",
        "--results-dir",
        str(results_root),
    ]
    subprocess.check_call(cmd)

    for dataset in ["synthetic_heston", "real_spy"]:
        for method in ["erm_baseline", "hirm_full"]:
            metrics_path = results_root / dataset / method / "seed_0" / "diagnostics.jsonl"
            assert metrics_path.exists(), f"Missing metrics output for {dataset}-{method}"
            contents = metrics_path.read_text(encoding="utf-8").strip().splitlines()
            assert contents, "Diagnostics file should not be empty"
