import subprocess
import subprocess
import sys
from pathlib import Path


def test_ablation_study_and_analysis(tmp_path):
    results_root = tmp_path / "ablation_study"

    cmd = [
        sys.executable,
        "scripts/run_grid.py",
        "--config",
        "configs/experiments/ablation_study.yaml",
        "--mode",
        "ablation",
        "--datasets",
        "synthetic_heston",
        "--ablations",
        "hirm_full,erm_baseline",
        "--seeds",
        "0",
        "--device",
        "cpu",
        "--reduced",
        "--results-dir",
        str(results_root),
    ]
    subprocess.check_call(cmd)

    diagnostics_path = results_root / "synthetic_heston" / "hirm_full" / "seed_0" / "diagnostics.jsonl"
    assert diagnostics_path.exists(), "Diagnostics output missing"

    analysis_out = tmp_path / "summary.csv"
    subprocess.check_call(
        [
            sys.executable,
            "analysis/analyze_ablation.py",
            "--root_dir",
            str(results_root),
            "--output",
            str(analysis_out),
        ]
    )

    assert analysis_out.exists(), "Analysis summary not written"
    with analysis_out.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle.readlines() if line.strip()]
    assert len(lines) > 1, "Summary CSV should contain at least one row"
