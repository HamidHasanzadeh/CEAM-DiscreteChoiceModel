"""CMA-ES search: parameter mapping, selection, and reproducibility tests."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ceam.optimization import SearchSpace, optimize_hyperparameters


def test_decode_center_and_boundaries():
    s = SearchSpace(
        learning_rate=(1e-5, 1e-3),
        epochs=(10, 30),
        batch_size=(8, 32),
        correction_penalty=(1e-8, 1e-2),
        hidden_units=(4, 12),
    )
    low = s.decode(np.zeros(5), seed=42, penalty_form="norm")
    high = s.decode(np.ones(5), seed=42, penalty_form="squared")
    mid = s.decode(np.full(5, .5), seed=42, penalty_form="norm")
    assert low.learning_rate == pytest.approx(1e-5)
    assert high.learning_rate == pytest.approx(1e-3)
    assert low.epochs == 10 and high.epochs == 30
    assert low.batch_size == 8 and high.batch_size == 32
    assert mid.learning_rate == pytest.approx(1e-4)
    assert mid.correction_penalty == pytest.approx(1e-5)
    assert mid.epochs == 20 and mid.hidden_units == 8
    assert high.penalty_form == "squared"


def test_cmaes_tunes_validation_loss_without_using_test(monkeypatch):
    """Mock fitting so the optimizer's orchestration is isolated and fast."""
    calls = []

    def mock_fit(train, validation, config):
        calls.append((train, validation, config))
        # Smooth deterministic objective with minimum in the search interior.
        loss = (np.log10(config.learning_rate) + 3.5) ** 2
        loss += (config.hidden_units - 7) ** 2 / 100
        return object(), {"best_validation_log_loss": float(loss)}

    monkeypatch.setattr("ceam.optimization.fit_model", mock_fit)
    result = optimize_hyperparameters(
        object(), object(),
        space=SearchSpace(
            learning_rate=(1e-5, 1e-2), epochs=(2, 4),
            batch_size=(4, 8), correction_penalty=(1e-9, 1e-4),
            hidden_units=(4, 10),
        ),
        seed=10, generations=2, population_size=4, repeats=2,
    )
    assert result["optimizer"] == "CMA-ES"
    assert len(result["evaluations"]) == 8
    assert result["best"]["validation_log_loss"] == min(
        rec["validation_log_loss"] for rec in result["evaluations"]
    )
    assert len(calls) >= 2
    assert {call[2].seed for call in calls} == {10, 11}
    assert result["unique_configurations_evaluated"] <= len(result["evaluations"])


def test_invalid_candidate_and_budget():
    with pytest.raises(ValueError):
        SearchSpace().decode(np.ones(3), seed=3, penalty_form="norm")
    with pytest.raises(ValueError):
        optimize_hyperparameters(None, None, generations=0)
