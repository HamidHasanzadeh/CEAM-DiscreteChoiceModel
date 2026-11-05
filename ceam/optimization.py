"""CMA-ES optimization of CEAM training hyperparameters on SwissMetro.

CMA-ES searches a normalized five-dimensional parameter space and minimizes
validation cross entropy. The held-out test partition is never used for
candidate ranking. 
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log
from typing import Callable

import numpy as np

from .data import Split
from .training import TrainingConfig, fit_model


@dataclass(frozen=True)
class SearchSpace:
    """Bounds for [lr, epochs, batch size, lambda, hidden neurons]."""

    learning_rate: tuple[float, float] = (1e-5, 1e-2)
    epochs: tuple[int, int] = (20, 150)
    batch_size: tuple[int, int] = (8, 128)
    correction_penalty: tuple[float, float] = (1e-10, 1e-1)
    hidden_units: tuple[int, int] = (4, 64)

    def __post_init__(self) -> None:
        for name in ("learning_rate", "correction_penalty"):
            low, high = getattr(self, name)
            if not (0 < low < high):
                raise ValueError(f"{name} requires positive, ordered bounds")
        for name in ("epochs", "batch_size", "hidden_units"):
            low, high = getattr(self, name)
            if not (1 <= low < high):
                raise ValueError(f"{name} requires positive, ordered bounds")

    def decode(self, candidate: np.ndarray, *, seed: int, penalty_form: str) -> TrainingConfig:
        """Decode coordinates in [0, 1], log-scale for positive scale parameters."""
        x = np.asarray(candidate, dtype=float)
        if x.shape != (5,) or not np.isfinite(x).all():
            raise ValueError("CMA-ES candidate must contain five finite parameters")
        x = np.clip(x, 0, 1)

        def logarithmic(pair: tuple[float, float], fraction: float) -> float:
            return float(exp(log(pair[0]) + fraction * (log(pair[1]) - log(pair[0]))))

        def integer(pair: tuple[int, int], fraction: float) -> int:
            return int(np.clip(round(pair[0] + fraction * (pair[1] - pair[0])), *pair))

        return TrainingConfig(
            seed=seed,
            learning_rate=logarithmic(self.learning_rate, x[0]),
            epochs=integer(self.epochs, x[1]),
            batch_size=integer(self.batch_size, x[2]),
            correction_penalty=logarithmic(self.correction_penalty, x[3]),
            hidden_units=integer(self.hidden_units, x[4]),
            penalty_form=penalty_form,
        )


def optimize_hyperparameters(
    train: Split,
    validation: Split,
    *,
    space: SearchSpace | None = None,
    seed: int = 42,
    generations: int = 8,
    population_size: int = 8,
    sigma: float = 0.22,
    penalty_form: str = "squared",
    repeats: int = 1,
    callback: Callable[[dict], None] | None = None,
) -> dict:
    """Run CMA-ES with validation log-loss as the minimized objective.

    Each candidate is retrained from scratch with identical fixed seed(s).
    Candidates that decode to identical configurations share evaluations.
    Rounding of integer parameters and boundary clipping are explicit.
    Returned best candidate is the best **evaluated** configuration, not
    necessarily the CMA-ES distribution mean.
    """
    if generations < 1 or population_size < 4 or repeats < 1 or sigma <= 0:
        raise ValueError("Require generations >= 1, population >= 4, repeats >= 1 and sigma > 0")
    if penalty_form not in {"norm", "squared"}:
        raise ValueError("Invalid penalty form")
    try:
        import cma
    except ImportError as exc:
        raise ImportError("Install the CMA-ES dependency with: pip install cma") from exc

    bounds = space or SearchSpace()
    optimizer = cma.CMAEvolutionStrategy(
        [0.5] * 5, sigma,
        {"bounds": [0.0, 1.0], "popsize": population_size, "seed": seed, "verbose": -9},
    )
    history: list[dict] = []
    cache: dict[tuple, tuple[float, list[float]]] = {}
    best: dict | None = None

    for generation in range(1, generations + 1):
        candidates = optimizer.ask()
        fitness: list[float] = []
        for index, candidate in enumerate(candidates):
            config = bounds.decode(np.asarray(candidate), seed=seed, penalty_form=penalty_form)
            key = (config.learning_rate, config.epochs, config.batch_size,
                   config.correction_penalty, config.hidden_units, config.penalty_form)
            cached = key in cache
            if not cached:
                scores: list[float] = []
                for repeat in range(repeats):
                    config_repeat = TrainingConfig(**{**asdict(config), "seed": seed + repeat})
                    _, training = fit_model(train, validation, config_repeat)
                    scores.append(float(training["best_validation_log_loss"]))
                value = float(np.mean(scores))
                if not np.isfinite(value):
                    value = 1e12
                cache[key] = (value, scores)
            value, scores = cache[key]
            fitness.append(value)
            record = {
                "generation": generation, "candidate": index + 1,
                "configuration": asdict(config),
                "validation_log_loss": value,
                "repeat_log_losses": scores,
                "cached": cached,
            }
            history.append(record)
            if best is None or value < best["validation_log_loss"]:
                best = {"configuration": asdict(config),
                        "validation_log_loss": value,
                        "generation": generation,
                        "candidate": index + 1}
            if callback:
                callback(record)
        optimizer.tell(candidates, fitness)
    assert best is not None
    return {
        "optimizer": "CMA-ES",
        "selection_metric": "mean best-checkpoint validation log-loss",
        "search_space": asdict(bounds),
        "seed": seed, "generations": generations,
        "population_size": population_size, "repeats_per_candidate": repeats,
        "best": best, "evaluations": history,
        "unique_configurations_evaluated": len(cache),
    }
