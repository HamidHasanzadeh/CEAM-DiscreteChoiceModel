"""Independent MNL reference and optimizer smoke test."""
import numpy as np
from benchmarks.mnl import evaluate, fit, utilities


def test_mnl_utility_order_and_reference_asc():
    x = np.array([[1, 2, 2, 3, 3, 4]], dtype=float)
    b = np.array([.5, 1., -2., -1.])
    np.testing.assert_allclose(utilities(x, b), [[-4., -6.5, -9.]])


def test_mnl_fit_returns_probabilistic_metrics():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(120, 6))
    y = rng.integers(0, 3, size=120)
    fitted = fit(x, y)
    result = evaluate(x, y, fitted.parameters)
    assert len(fitted.parameters) == 4
    assert result["log_loss"] >= 0
    assert 0 <= result["accuracy"] <= 1


def test_mnl_availability_mask():
    x = np.array([[1., 2., 2., 3., 3., 4.], [2., 3., 1., 1., 3., 2.]])
    available = np.array([[True, True, False], [False, True, True]])
    parameters = np.array([.5, 1., -2., -1.])
    masked = utilities(x, parameters, available)
    assert np.isneginf(masked[0, 2]) and np.isneginf(masked[1, 0])
    result = evaluate(x, np.array([0, 2]), parameters, available)
    assert np.isfinite(result["log_loss"])
