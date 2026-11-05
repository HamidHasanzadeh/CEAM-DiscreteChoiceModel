"""Complete-model substitution and local economic-regularity diagnostics."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .model import CEAM

# Features [TRAIN_TT, TRAIN_COST, SM_TT, SM_COST, CAR_TT, CAR_COST].
FEATURE_NAMES = ("Train time", "Train cost", "SwissMetro time", "SwissMetro cost", "Car time", "Car cost")


def shock_response(model: CEAM, features: Tensor, feature_index: int,
                   relative_change: float, availability: Tensor | None = None) -> dict[str, list[float]]:
    """Compute changes in average *complete-model* choice probabilities.

    Values are percentage points. This is a conditional model prediction, not
    an identified causal effect or a guaranteed welfare measure.
    """
    if feature_index not in range(6):
        raise ValueError("feature_index must range from 0 to 5")
    modified = features.detach().clone()
    modified[:, feature_index] *= 1.0 + relative_change
    model.eval()
    with torch.no_grad():
        baseline = model.probabilities(features, availability).mean(dim=0)
        shocked = model.probabilities(modified, availability).mean(dim=0)
    return {"baseline_share_percent": (100 * baseline).tolist(),
            "shocked_share_percent": (100 * shocked).tolist(),
            "change_percentage_points": (100 * (shocked - baseline)).tolist()}


def own_probability_derivatives(model: CEAM, features: Tensor,
                                availability: Tensor | None = None) -> Tensor:
    """d P(alternative) / d own time and own cost using autograd.

    Returns shape [observations, 3 alternatives, 2 attributes].
    Availability restrictions, when relevant, must be applied upstream.
    """
    model.eval()
    x = features.detach().clone().requires_grad_(True)
    p = model.probabilities(x, availability)
    pairs = ((0, 1), (2, 3), (4, 5))
    per_mode = []
    for alt, (time_idx, cost_idx) in enumerate(pairs):
        gradient = torch.autograd.grad(p[:, alt].sum(), x, retain_graph=alt < 2)[0]
        per_mode.append(torch.stack((gradient[:, time_idx], gradient[:, cost_idx]), dim=1))
    return torch.stack(per_mode, dim=1).detach()


def violation_percentages(derivatives: Tensor, tolerance: float = 1e-7,
                          availability: Tensor | None = None) -> dict[str, dict[str, float]]:
    """Own-probability sign-departure rates, among eligible observations only.

    Supply the same availability mask used for model evaluation to exclude
    unavailable alternatives from each alternative-specific denominator.
    Without a mask, all observations are assumed eligible (legacy behavior).
    """
    if derivatives.ndim != 3 or derivatives.shape[1:] != (3, 2):
        raise ValueError("Expected derivatives with shape [n, 3, 2]")
    if availability is None:
        availability = torch.ones(derivatives.shape[:2], device=derivatives.device, dtype=torch.bool)
    elif availability.shape != derivatives.shape[:2] or availability.dtype != torch.bool:
        raise ValueError("availability must be a Boolean [n, 3] mask")
    if not availability.any(dim=0).all():
        raise ValueError("Each alternative needs at least one eligible observation")
    return {name: {att: float((derivatives[availability[:, i], i, j] > tolerance).float().mean() * 100)
                   for j, att in enumerate(("time", "cost"))}
            for i, name in enumerate(("Train", "SwissMetro", "Car"))}


def relative_utility_magnitude(model: CEAM, features: Tensor) -> float:
    """Descriptive share of centered neural-vs-linear score magnitudes.

    This diagnostic uses per-observation alternative-mean centering and L2
    magnitudes. This describes relative score magnitudes, not causal attribution.
    """
    with torch.no_grad():
        comp = model.components(features)
        v = comp.linear_utility - comp.linear_utility.mean(dim=1, keepdim=True)
        d = comp.neural_correction - comp.neural_correction.mean(dim=1, keepdim=True)
        v_mag = torch.linalg.vector_norm(v, dim=1).mean()
        d_mag = torch.linalg.vector_norm(d, dim=1).mean()
        return float(d_mag / (d_mag + v_mag).clamp_min(1e-12))
