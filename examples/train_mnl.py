"""Fit an independent SwissMetro MNL using identical data partitions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.mnl import evaluate, fit
from ceam.data import load_swissmetro, partition


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent SwissMetro MNL baseline")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--respondent-column", default="ID")
    parser.add_argument("--output", type=Path, default=Path("results/generated/swissmetro/mnl.json"))
    args = parser.parse_args()
    parts = partition(load_swissmetro(args.data), seed=args.seed,
                      group_column=args.respondent_column)
    fitted = fit(parts.train.features.numpy(), parts.train.labels.numpy(),
                 parts.train.availability.numpy())
    metrics = {name: evaluate(getattr(parts, name).features.numpy(),
                              getattr(parts, name).labels.numpy(), fitted.parameters,
                              getattr(parts, name).availability.numpy())
               for name in ("train", "validation", "test")}
    report = {"model": "Independent MNL", "parameters": dict(zip(
        ("ASC_SwissMetro", "ASC_Car", "beta_time", "beta_cost"),
        fitted.parameters.tolist())), "converged": fitted.converged,
        "optimizer_message": fitted.message, "metrics": metrics,
        "splitting": "respondent-disjoint" if args.respondent_column else "row-wise"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"test": metrics["test"], "converged": fitted.converged}, indent=2))


if __name__ == "__main__":
    main()
