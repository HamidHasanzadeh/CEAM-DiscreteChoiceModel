"""SwissMetro CEAM: explicit utility plus learned cross-alternative correction.

Alternative order: Train, SwissMetro, Car.
Feature order: train time, train cost, metro time, metro cost, car time, car cost.
Features must use the same numerical scaling in training and evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Components:
    """Deterministic linear utility, neural adjustment, and combined score."""

    linear_utility: Tensor
    neural_correction: Tensor
    logits: Tensor


class UtilityLayer(nn.Module):
    """Four-parameter shared time/cost SwissMetro utility specification.

    Train is the reference alternative with its ASC normalized to zero.
    Parameters are unconstrained.
    """

    def __init__(self) -> None:
        super().__init__()
        self.asc_metro = nn.Parameter(torch.zeros(()))
        self.asc_car = nn.Parameter(torch.zeros(()))
        self.beta_time = nn.Parameter(torch.zeros(()))
        self.beta_cost = nn.Parameter(torch.zeros(()))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != 6:
            raise ValueError("Expected a matrix with six SwissMetro features per observation.")
        time = features[:, [0, 2, 4]]
        cost = features[:, [1, 3, 5]]
        asc = torch.stack((torch.zeros_like(self.asc_metro), self.asc_metro, self.asc_car))
        return self.beta_time * time + self.beta_cost * cost + asc


class CEAM(nn.Module):
    """Linear utility augmented by a network of all attributes and utilities.

    The neural path receives the concatenation [X, V] and has exactly one
    fully connected hidden layer with ReLU, followed by a linear output
    projection. The three correction outputs are added to linear utilities. 
    """

    def __init__(self, hidden_units: int = 6) -> None:
        super().__init__()
        if hidden_units <= 0:
            raise ValueError("hidden_units must be positive")
        self.utility_layer = UtilityLayer()
        self.cross_effect = nn.Sequential(nn.Linear(9, hidden_units), nn.ReLU(), nn.Linear(hidden_units, 3))

    def components(self, features: Tensor) -> Components:
        linear = self.utility_layer(features)
        correction = self.cross_effect(torch.cat((features, linear), dim=1))
        return Components(linear, correction, linear + correction)

    @staticmethod
    def mask_logits(logits: Tensor, availability: Tensor | None) -> Tensor:
        if availability is None:
            return logits
        if availability.shape != logits.shape or availability.dtype != torch.bool:
            raise ValueError("availability must be a Boolean matrix matching logits")
        if not availability.any(dim=1).all():
            raise ValueError("Every observation must have at least one available alternative")
        return logits.masked_fill(~availability, -torch.inf)

    def forward(self, features: Tensor, availability: Tensor | None = None) -> Tensor:
        """Return availability-masked logits for cross-entropy loss."""
        return self.mask_logits(self.components(features).logits, availability)

    def probabilities(self, features: Tensor, availability: Tensor | None = None) -> Tensor:
        return F.softmax(self(features, availability), dim=1)

    def linear_probabilities(self, features: Tensor, availability: Tensor | None = None) -> Tensor:
        return F.softmax(self.mask_logits(self.utility_layer(features), availability), dim=1)

    def coefficient_summary(self) -> dict[str, float]:
        layer = self.utility_layer
        return {name: float(value.detach().cpu()) for name, value in {
            "ASC_SwissMetro": layer.asc_metro,
            "ASC_Car": layer.asc_car,
            "beta_time": layer.beta_time,
            "beta_cost": layer.beta_cost,
        }.items()}
