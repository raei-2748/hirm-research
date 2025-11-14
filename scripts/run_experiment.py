"""Minimal experiment runner for Phase 1."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hirm.envs import build_env
from hirm.utils import config as config_utils
from hirm.utils.logging import ExperimentLogger
from hirm.utils.seed import set_seed
from hirm.utils.serialization import save_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/tiny_test.yaml",
        help="Path to experiment config",
    )
    return parser.parse_args()


def run_episode(env_config: Dict[str, Any], seed: int, output_dir: Path, logger: ExperimentLogger) -> None:
    env = build_env(env_config)
    split = env_config.get("split", "train")
    episodes_to_sample = int(env_config.get("episodes", 1))
    summaries = []
    for episode_idx in range(episodes_to_sample):
        episode = env.sample_episode(split=split)
        logger.info(
            f"episode={episode_idx} split={split} env_id={episode.env_id} horizon={len(episode.prices) - 1}"
        )
        save_dict(
            {
                "prices": episode.prices,
                "states": episode.states,
                "pnl": episode.pnl,
                "env_id": episode.env_id,
                "meta": episode.meta,
            },
            output_dir / f"episode_{episode_idx}.json",
        )
        summaries.append({"env_id": episode.env_id, "split": split})
    save_dict({"metrics": {"episodes": summaries}}, output_dir / "metrics.json")


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
    run_episode(resolved_config["env"], seed, output_dir, logger)
    save_dict(resolved_config, output_dir / "resolved_config.json")
    logger.info("Experiment finished")
    logger.close()


if __name__ == "__main__":
    main()
