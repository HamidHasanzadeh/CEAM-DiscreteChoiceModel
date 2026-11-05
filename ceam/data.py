"""SwissMetro preprocessing and reproducible dataset partitioning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

FEATURE_COLUMNS = (
    "TRAIN_TT_SCALED", "TRAIN_COST_SCALED",
    "SM_TT_SCALED", "SM_COST_SCALED",
    "CAR_TT_SCALED", "CAR_CO_SCALED",
)
AVAILABILITY_COLUMNS = ("TRAIN_AV", "SM_AV", "CAR_AV")
REQUIRED_COLUMNS = (*FEATURE_COLUMNS, *AVAILABILITY_COLUMNS, "PURPOSE", "CHOICE")


@dataclass(frozen=True)
class Split:
    features: Tensor
    labels: Tensor
    availability: Tensor
    row_indices: np.ndarray


@dataclass(frozen=True)
class Partitions:
    train: Split
    validation: Split
    test: Split


def load_swissmetro(path: str | Path) -> pd.DataFrame:
    """Filter eligible choices and select pre-scaled SwissMetro attributes.

    Scaling is supplied by the input dataset, not estimated here.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SwissMetro input not found: {path}")
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required SwissMetro columns: {sorted(missing)}")
    df = df.loc[df["PURPOSE"].isin([1, 3]) & df["CHOICE"].isin([1, 2, 3])].copy()
    df = df.dropna(subset=[*FEATURE_COLUMNS, *AVAILABILITY_COLUMNS, "CHOICE"])
    if not df.loc[:, AVAILABILITY_COLUMNS].isin([0, 1]).all().all():
        raise ValueError("Availability indicators must be binary (0 or 1)")
    chosen = df["CHOICE"].to_numpy(dtype=np.int64) - 1
    available = df.loc[:, AVAILABILITY_COLUMNS].to_numpy(dtype=bool)
    if not available.any(axis=1).all() or not available[np.arange(len(df)), chosen].all():
        raise ValueError("Each observation needs an available chosen alternative")
    if df.empty:
        raise ValueError("No eligible SwissMetro observations after filtering")
    return df


def partition(df: pd.DataFrame, seed: int = 42, test_fraction: float = 0.15,
              validation_fraction: float = 0.15, group_column: str | None = "ID") -> Partitions:
    """Create reproducible splits; supply group_column to separate respondents.

    Defaults to respondent-disjoint splitting using the SwissMetro ID column.
    Pass group_column=None only to opt into row-wise splitting.
    """
    if test_fraction <= 0 or validation_fraction <= 0 or test_fraction + validation_fraction >= 1:
        raise ValueError("Expected positive test/validation fractions summing below one")
    rng = np.random.default_rng(seed)
    if group_column:
        if group_column not in df:
            raise ValueError(f"Missing respondent identifier: {group_column}")
        groups = df[group_column].to_numpy()
        unique = rng.permutation(pd.unique(groups))
        n_test = max(1, round(len(unique) * test_fraction))
        n_val = max(1, round(len(unique) * validation_fraction))
        if n_test + n_val >= len(unique):
            raise ValueError("Not enough respondent groups for a disjoint three-way split")
        test_groups = unique[:n_test]
        val_groups = unique[n_test:n_test+n_val]
        indices = np.arange(len(df))
        test_idx = indices[np.isin(groups, test_groups)]
        val_idx = indices[np.isin(groups, val_groups)]
        train_idx = indices[~np.isin(groups, np.concatenate((test_groups, val_groups)))]
    else:
        indices = rng.permutation(len(df))
        n_test = max(1, round(len(df) * test_fraction))
        n_val = max(1, round(len(df) * validation_fraction))
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test+n_val]
        train_idx = indices[n_test+n_val:]
    if not min(len(train_idx), len(val_idx), len(test_idx)):
        raise ValueError("Empty split: increase the number of observations")
    x = torch.as_tensor(df.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32).copy())
    y = torch.as_tensor(df["CHOICE"].to_numpy(dtype=np.int64).copy() - 1)
    availability = torch.as_tensor(df.loc[:, AVAILABILITY_COLUMNS].to_numpy(dtype=bool).copy())
    if not availability.any(dim=1).all() or not availability[torch.arange(len(y)), y].all():
        raise ValueError("Each observation needs an available chosen alternative")
    def subset(idx: np.ndarray) -> Split:
        return Split(x[idx], y[idx], availability[idx], idx)
    return Partitions(subset(train_idx), subset(val_idx), subset(test_idx))
