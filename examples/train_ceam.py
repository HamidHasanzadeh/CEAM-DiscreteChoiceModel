"""SwissMetro CEAM command line training and evaluation.

Run from repository root:
  python -m examples.train_ceam --data path/to/swissmetro_prepared.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ceam.data import load_swissmetro, partition
from ceam.training import TrainingConfig, evaluate, fit_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate CEAM on SwissMetro")
    parser.add_argument("--data", type=Path, required=True, help="CSV/XLSX with the six scaled fields, PURPOSE, CHOICE")
    parser.add_argument("--output", type=Path, default=Path("results/generated/swissmetro"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-units", type=int, default=6)
    parser.add_argument("--hidden-layers", type=str, default=None, help="Comma-separated widths, e.g. 32,16")
    parser.add_argument("--activation", choices=("relu", "tanh", "gelu"), default="relu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lambda-correction", type=float, default=1e-8)
    parser.add_argument("--penalty-form", choices=("norm", "squared"), default="squared")
    parser.add_argument("--penalty-reduction", choices=("mean", "sum"), default="mean")
    parser.add_argument("--respondent-column", default="ID", help="Respondent identifier (default: ID)")
    args = parser.parse_args()
    try:
        layers = None if args.hidden_layers is None else tuple(int(v.strip()) for v in args.hidden_layers.split(","))
    except ValueError:
        parser.error("--hidden-layers requires comma-separated integers")
    if layers is not None and (not layers or any(width <= 0 for width in layers)):
        parser.error("--hidden-layers requires positive integers")
    config = TrainingConfig(seed=args.seed, hidden_units=args.hidden_units,
                            hidden_layers=layers, activation=args.activation,
                            epochs=args.epochs, learning_rate=args.learning_rate,
                            batch_size=args.batch_size, correction_penalty=args.lambda_correction,
                            penalty_form=args.penalty_form,
                            penalty_reduction=args.penalty_reduction)
    dataframe = load_swissmetro(args.data)
    partitions = partition(dataframe, seed=args.seed, group_column=args.respondent_column)
    model, fit = fit_model(partitions.train, partitions.validation, config)
    metrics = {name: evaluate(model, getattr(partitions, name))
               for name in ("train", "validation", "test")}
    result = {"experiment": "SwissMetro / CEAM", "training": fit,
              "evaluation": metrics, "linear_coefficients": model.coefficient_summary(),
              "n_filtered": len(dataframe),
              "split_type": "respondent-disjoint" if args.respondent_column else "row-wise"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "config": fit["configuration"]}, args.output / "ceam.pt")
    print(json.dumps({"test": metrics["test"], "linear_coefficients": result["linear_coefficients"],
                      "output_directory": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
