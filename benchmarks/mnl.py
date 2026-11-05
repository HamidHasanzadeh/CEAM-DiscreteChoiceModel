"""Independently estimated four-parameter SwissMetro multinomial logit.

Alternatives: Train (reference ASC), SwissMetro, Car.
Uses the same six scaled input features as the CEAM utility layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


@dataclass(frozen=True)
class MNLFit:
    parameters: np.ndarray
    converged: bool
    message: str
    training_log_likelihood: float


def utilities(x: np.ndarray, parameters: np.ndarray, availability: np.ndarray | None = None) -> np.ndarray:
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError("Expected six scaled SwissMetro attributes")
    a_sm, a_car, time_beta, cost_beta = parameters
    time = x[:, [0, 2, 4]]
    cost = x[:, [1, 3, 5]]
    logits = time_beta * time + cost_beta * cost + np.array([0.0, a_sm, a_car])
    if availability is not None:
        availability = np.asarray(availability, dtype=bool)
        if availability.shape != logits.shape or not availability.any(axis=1).all():
            raise ValueError("Invalid availability matrix")
        logits = np.where(availability, logits, -np.inf)
    return logits


def fit(x: np.ndarray, y: np.ndarray, availability: np.ndarray | None = None) -> MNLFit:
    """Maximum-likelihood fit using BFGS and an analytical gradient."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if not len(x) or len(x) != len(y) or np.any((y < 0) | (y > 2)):
        raise ValueError("Invalid SwissMetro feature/choice observations")
    if availability is not None:
        availability = np.asarray(availability, dtype=bool)
        if availability.shape != (len(y), 3) or not availability[np.arange(len(y)), y].all():
            raise ValueError("Invalid availability or unavailable chosen alternative")
    time = x[:, [0, 2, 4]]
    cost = x[:, [1, 3, 5]]
    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = utilities(x, parameters, availability)
        p = softmax(logits, axis=1)
        error = p.copy()
        error[np.arange(len(y)), y] -= 1.0
        nll = float((logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]).sum())
        gradient = np.array([
            error[:, 1].sum(), error[:, 2].sum(),
            np.sum(error * time), np.sum(error * cost),
        ])
        return nll, gradient
    result = minimize(objective, np.zeros(4), method="BFGS", jac=True,
                      options={"maxiter": 500, "gtol": 1e-6})
    return MNLFit(parameters=np.asarray(result.x), converged=bool(result.success),
                  message=str(result.message), training_log_likelihood=-float(result.fun))


def evaluate(x: np.ndarray, y: np.ndarray, parameters: np.ndarray,
             availability: np.ndarray | None = None) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    logits = utilities(x, parameters, availability)
    if not np.isfinite(logits[np.arange(len(y)), y]).all():
        raise ValueError("Chosen alternative is unavailable")
    log_prob = logits - logsumexp(logits, axis=1, keepdims=True)
    chosen = log_prob[np.arange(len(y)), y]
    return {"n": len(y), "log_likelihood": float(chosen.sum()),
            "log_loss": float(-chosen.mean()),
            "accuracy": float(np.mean(logits.argmax(axis=1) == y))}
