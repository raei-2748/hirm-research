import subprocess
import sys
from pathlib import Path


def test_phase9_smoke(tmp_path):
    results_root = Path("results/phase9")

    cmd = [
        sys.executable,
        "scripts/run_experiment_grid.py",
        "--config",
        "configs/experiments/phase9.yaml",
        "--datasets",
        "synthetic_heston,real_spy",
        "--methods",
        "erm,hirm",
        "--seeds",
        "0",
        "--device",
        "cpu",
        "--reduced",
    ]
    subprocess.check_call(cmd)

    for dataset in ["synthetic_heston", "real_spy"]:
        for method in ["erm", "hirm"]:
            metrics_path = results_root / dataset / method / "seed_0" / "diagnostics.jsonl"
            assert metrics_path.exists(), f"Missing metrics output for {dataset}-{method}"
            contents = metrics_path.read_text(encoding="utf-8").strip().splitlines()
            assert contents, "Diagnostics file should not be empty"
