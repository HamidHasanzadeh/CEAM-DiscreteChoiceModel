"""Focused tests for the SwissMetro CEAM equations and diagnostics."""
import numpy as np
import pandas as pd
import pytest
import torch

from ceam.data import FEATURE_COLUMNS, partition
from ceam.diagnostics import own_probability_derivatives, shock_response, violation_percentages
from ceam.model import CEAM, UtilityLayer
from ceam.training import TrainingConfig, evaluate, fit_model


def fake_data(n=36):
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(rng.normal(size=(n, 6)), columns=FEATURE_COLUMNS)
    frame["CHOICE"] = np.tile([1, 2, 3], n // 3)
    frame["PURPOSE"] = 1
    frame["ID"] = np.repeat(np.arange(n // 3), 3)
    for column in ("TRAIN_AV", "SM_AV", "CAR_AV"):
        frame[column] = 1
    frame["respondent"] = np.repeat(np.arange(n // 3), 3)
    return frame


def test_utility_layer_train_reference_and_shared_coefficients():
    layer = UtilityLayer()
    with torch.no_grad():
        layer.beta_time.fill_(-2)
        layer.beta_cost.fill_(-1)
        layer.asc_metro.fill_(0.5)
        layer.asc_car.fill_(1)
    features = torch.tensor([[1., 2., 2., 3., 3., 4.]])
    assert torch.allclose(layer(features), torch.tensor([[-4., -6.5, -9.]]))


def test_ceam_linear_plus_neural_scores_and_probabilities():
    model = CEAM(hidden_units=5)
    x = torch.randn(8, 6)
    c = model.components(x)
    assert c.logits.shape == (8, 3)
    assert torch.allclose(c.logits, c.linear_utility + c.neural_correction)
    assert torch.allclose(model.probabilities(x).sum(dim=1), torch.ones(8))


def test_respondent_disjoint_splits():
    frame = fake_data()
    parts = partition(frame, group_column="respondent", seed=42)
    sets = [set(frame.iloc[p.row_indices]["respondent"]) for p in
            (parts.train, parts.validation, parts.test)]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])


def test_train_evaluate_and_complete_model_derivatives():
    parts = partition(fake_data(), group_column="respondent")
    model, fit = fit_model(parts.train, parts.validation,
                           TrainingConfig(epochs=2, batch_size=6, hidden_units=4))
    assert len(fit["history"]) == 2
    metrics = evaluate(model, parts.test)
    assert 0 <= metrics["accuracy"] <= 1
    derivatives = own_probability_derivatives(model, parts.test.features, parts.test.availability)
    assert derivatives.shape == (len(parts.test.labels), 3, 2)
    rates = violation_percentages(derivatives, availability=parts.test.availability)
    assert set(rates) == {"Train", "SwissMetro", "Car"}
    response = shock_response(model, parts.test.features, feature_index=1, relative_change=.10,
                              availability=parts.test.availability)
    assert abs(sum(response["change_percentage_points"])) < 1e-4


def test_invalid_swissmetro_shape():
    with pytest.raises(ValueError):
        CEAM()(torch.zeros(4, 5))


def test_default_regularization_is_manuscript_squared_l2():
    """The manuscript penalty is lambda times the squared L2 norm of delta."""
    cfg = TrainingConfig()
    assert cfg.penalty_form == "squared"
    correction = torch.tensor([[3.0, 4.0, 0.0], [0.0, -2.0, 0.0]])
    squared_l2 = torch.linalg.vector_norm(correction).square()
    assert squared_l2.item() == pytest.approx(29.0)
    assert (cfg.correction_penalty * squared_l2).item() == pytest.approx(29.0e-8)


def test_availability_mask_sets_unavailable_probability_to_zero():
    model = CEAM(hidden_units=4)
    x = torch.randn(3, 6)
    availability = torch.tensor([[True, True, False], [False, True, True], [True, False, True]])
    probs = model.probabilities(x, availability)
    assert torch.equal(probs[~availability], torch.zeros_like(probs[~availability]))
    assert torch.allclose(probs.sum(dim=1), torch.ones(3))


def test_default_partition_is_respondent_disjoint():
    frame = fake_data()
    parts = partition(frame)
    groups = [set(frame.iloc[p.row_indices]["ID"]) for p in (parts.train, parts.validation, parts.test)]
    assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i + 1, 3))


def test_unavailable_chosen_alternative_is_rejected():
    frame = fake_data()
    frame.loc[0, "TRAIN_AV"] = 0
    with pytest.raises(ValueError, match="available chosen"):
        partition(frame)


def test_regularization_penalty_is_batch_summed_not_mean_normalized():
    """Lock in the published reference-implementation convention."""
    single = torch.tensor([[3.0, 4.0, 0.0]])
    doubled = single.repeat(2, 1)
    assert torch.linalg.vector_norm(single).square().item() == pytest.approx(25.0)
    assert torch.linalg.vector_norm(doubled).square().item() == pytest.approx(50.0)


def test_violation_rates_use_eligible_observations():
    derivatives = torch.zeros(4, 3, 2)
    derivatives[0, 2, 0] = 1.0
    derivatives[1, 2, 0] = 1.0  # unavailable Car: must not count
    derivatives[2, 2, 0] = -1.0
    derivatives[3, 2, 0] = 1.0  # unavailable Car: must not count
    availability = torch.tensor([
        [True, True, True],
        [True, True, False],
        [True, True, True],
        [True, True, False],
    ])
    rates = violation_percentages(derivatives, availability=availability)
    assert rates["Car"]["time"] == pytest.approx(50.0)
    assert violation_percentages(derivatives)["Car"]["time"] == pytest.approx(75.0)
    with pytest.raises(ValueError, match="Boolean"):
        violation_percentages(derivatives, availability=availability.int())
