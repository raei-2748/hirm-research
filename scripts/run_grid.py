"""Universal grid runner for HIRM research."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hirm.engine import train_and_evaluate
from hirm.experiments.ablations import apply_ablation_to_config, get_ablation_config, list_ablations
from hirm.experiments.registry import ExperimentRunConfig
from hirm.utils.config import ConfigNode, load_config, to_plain_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base config")
    parser.add_argument("--mode", choices=["benchmark", "ablation"], default="benchmark")

    parser.add_argument("--datasets", type=str, help="Comma-split datasets")
    parser.add_argument("--methods", type=str, help="Comma-split methods (benchmark only)")
    parser.add_argument("--ablations", type=str, help="Comma-split ablations (ablation only)")
    parser.add_argument("--seeds", type=str, default="0,1,2")

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-dir", default="results/custom")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reduced", action="store_true", help="Smoke test mode")
    return parser.parse_args()


def main():
    args = parse_args()
    base_cfg = load_config(args.config)
    device = torch.device(args.device)
    root = Path(args.results_dir)

    datasets = (args.datasets or "synthetic_heston,real_spy").split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    if args.reduced:
        seeds = seeds[:1]

    tasks = []
    if args.mode == "benchmark":
        methods = (args.methods or "erm,hirm").split(",")
        for dataset in datasets:
            for method in methods:
                for seed in seeds:
                    tasks.append((dataset, method, seed, None))
    else:
        ablations = (args.ablations or ",".join(list_ablations())).split(",")
        for dataset in datasets:
            for ablation_name in ablations:
                for seed in seeds:
                    tasks.append((dataset, ablation_name, seed, get_ablation_config(ablation_name)))

    print(f"Found {len(tasks)} experiments to run in {args.mode} mode.")

    for dataset, method_or_ablation, seed, ablation_obj in tasks:
        method = ablation_obj.method if ablation_obj else method_or_ablation
        name = ablation_obj.name if ablation_obj else method

        cfg = ConfigNode(to_plain_dict(base_cfg))
        if ablation_obj:
            cfg = apply_ablation_to_config(cfg, ablation_obj)
        else:
            cfg.objective.name = method
        cfg.seed = seed

        out_dir = root / dataset / name / f"seed_{seed}"
        run_cfg = ExperimentRunConfig(
            dataset=dataset,
            method=method,
            seed=seed,
            config=cfg,
            device=device,
            ablation=ablation_obj,
        )

        print(f"--- Starting: {dataset} / {name} / seed {seed} ---")
        train_and_evaluate(run_cfg, out_dir, force_rerun=args.force)


if __name__ == "__main__":
    main()
