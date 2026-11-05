"""Focused tests for the SwissMetro CEAM equations and diagnostics."""
import numpy as np
import pandas as pd
import pytest
import torch

from ceam.data import FEATURE_COLUMNS, partition
from ceam.diagnostics import own_probability_derivatives, shock_response, violation_percentages
from ceam.model import CEAM, UtilityLayer
from ceam.training import TrainingConfig, correction_penalty_value, evaluate, fit_model


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
    normalized = correction_penalty_value(correction)
    assert cfg.penalty_reduction == "mean"
    assert normalized.item() == pytest.approx(14.5)
    assert (cfg.correction_penalty * normalized).item() == pytest.approx(14.5e-8)


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


def test_normalized_penalty_is_invariant_to_batch_repetition():
    single = torch.tensor([[3.0, 4.0, 0.0]])
    doubled = single.repeat(2, 1)
    assert correction_penalty_value(single).item() == pytest.approx(25.0)
    assert correction_penalty_value(doubled).item() == pytest.approx(25.0)
    assert correction_penalty_value(doubled, reduction="sum").item() == pytest.approx(50.0)
    assert correction_penalty_value(doubled, penalty_form="norm", reduction="sum").item() == pytest.approx(50.0 ** .5)


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


@pytest.mark.parametrize("widths,activation", [((5,), "relu"), ((16, 8), "tanh"), ((12, 8, 4), "gelu")])
def test_configurable_neural_depth_and_activation(widths, activation):
    model = CEAM(hidden_layers=widths, activation=activation)
    x = torch.randn(7, 6)
    comp = model.components(x)
    assert comp.neural_correction.shape == (7, 3)
    assert len([layer for layer in model.cross_effect if isinstance(layer, torch.nn.Linear)]) == len(widths) + 1
    assert torch.allclose(model.probabilities(x).sum(dim=1), torch.ones(7))


def test_legacy_one_hidden_layer_has_identical_parameter_structure():
    model = CEAM(hidden_units=6)
    assert list(model.cross_effect.state_dict()) == [
        "0.weight", "0.bias", "2.weight", "2.bias"
    ]


@pytest.mark.parametrize("invalid", [(), (3, 0), (2, -1), (3, 2.5), "16,8"])
def test_invalid_hidden_architecture_is_rejected(invalid):
    with pytest.raises(ValueError):
        CEAM(hidden_layers=invalid)


def test_invalid_activation_and_penalty_reduction_are_rejected():
    with pytest.raises(ValueError):
        CEAM(hidden_layers=(4,), activation="other")
    with pytest.raises(ValueError):
        correction_penalty_value(torch.zeros(2, 3), reduction="median")


def test_training_with_multilayer_network():
    parts = partition(fake_data(), group_column="respondent")
    config = TrainingConfig(epochs=2, batch_size=6, hidden_layers=(12, 6), activation="gelu")
    model, result = fit_model(parts.train, parts.validation, config)
    assert model.hidden_layers == (12, 6)
    assert result["configuration"]["penalty_reduction"] == "mean"
    assert 0 <= evaluate(model, parts.test)["accuracy"] <= 1
