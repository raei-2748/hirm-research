"""Minimal experiment runner for Phase 1."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirm.utils import config as config_utils
from hirm.utils.logging import ExperimentLogger
from hirm.utils.seed import set_seed
from hirm.utils.serialization import save_dict
from hirm.train.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/tiny_test.yaml",
        help="Path to experiment config",
    )
    return parser.parse_args()


def run_training(config: Dict[str, Any], output_dir: Path, logger: ExperimentLogger) -> None:
    trainer = Trainer(
        env_config=config["env"],
        model_config=config["model"],
        objective_config=config["objective"],
        training_config=config.get("training", {}),
    )
    metrics = trainer.train()
    save_dict({"metrics": metrics}, output_dir / "metrics.json")
    logger.info(f"Training metrics: {metrics}")


def main() -> None:
    args = parse_args()
    resolved_config = config_utils.load_config(args.config)
    experiment_name = resolved_config["experiment"]["name"]
    output_dir = Path(resolved_config["logging"]["output_dir"]) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(experiment_name, resolved_config["logging"]["output_dir"])
    seed = int(resolved_config.get("seed", 0))
    set_seed(seed)
    logger.info(f"Loaded config from {args.config}")
    run_training(resolved_config, output_dir, logger)
    save_dict(resolved_config, output_dir / "resolved_config.json")
    logger.info("Experiment finished")
    logger.close()


if __name__ == "__main__":
    main()
