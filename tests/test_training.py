"""Training/calibration checks for leakage, determinism, and missing sensors."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from physiobench.data import generate_dataset, split_subjects
from physiobench.experiments import split_validation, stress
from physiobench.training import PhysioModel, TrainConfig, predict, seed_everything, train
from physiobench.posttraining import (fit_temperature, binary_metrics, acceptance_threshold,
                                     correctness, risk_coverage_curve, fit_subject_conformal)


@pytest.fixture(scope="module")
def small_splits():
    return split_subjects(generate_dataset(n_subjects=30, duration_s=10, seed=42), seed=7)


@pytest.mark.parametrize("architecture", ["cnn", "transformer"])
def test_prediction_uses_only_observations(architecture, small_splits):
    data = small_splits["test_id"].subset(np.arange(4))
    seed_everything(1)
    model = PhysioModel(architecture, data.fs, data.x.shape[-1])
    before = predict(model, data)
    counterfactual = copy.deepcopy(data)
    counterfactual.y[:] = -1000
    counterfactual.corruption[:] = ~counterfactual.corruption
    counterfactual.severity[:] = 1
    counterfactual.subject_id[:] = 99999
    counterfactual.scenario[:] = "test_oracle"
    counterfactual.latent_hr[:] = 12345
    counterfactual.latent_rr[:] = 54321
    after = predict(model, counterfactual)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_training_updates_weights_deterministically(small_splits):
    data = small_splits["train"].subset(np.arange(16))
    validation = small_splits["val"].subset(np.arange(4))
    config = TrainConfig(epochs=2, batch_size=8)
    models = []
    for _ in range(2):
        seed_everything(19)
        model = PhysioModel("cnn", data.fs, data.x.shape[-1])
        initial = model.regression.weight.detach().clone()
        report = train(model, data, validation, config, seed=19)
        assert not torch.equal(initial, model.regression.weight)
        assert np.isfinite(report["best_validation_score"])
        models.append(predict(model, validation))
    for key in models[0]:
        np.testing.assert_array_equal(models[0][key], models[1][key])


def test_calibration_subjects_are_disjoint(small_splits):
    tuning, calibration = split_validation(small_splits["val"], seed=22)
    assert not set(tuning.subject_id) & set(calibration.subject_id)
    assert set(tuning.subject_id) | set(calibration.subject_id) == set(small_splits["val"].subject_id)


@pytest.mark.parametrize("name,index", [("ecg", 0), ("ppg", 1), ("respiration", 2), ("motion", 3)])
def test_explicit_missing_modality_ablation(name, index, small_splits):
    source = small_splits["test_ood"]
    missing = stress(source, f"drop_{name}", seed=1)
    assert not missing.observed[:, index].any()
    assert (missing.x[:, index] == 0).all()
    np.testing.assert_array_equal(source.y, missing.y)
    np.testing.assert_array_equal(source.case_id, missing.case_id)
    other = [i for i in range(4) if i != index]
    np.testing.assert_array_equal(source.x[:, other], missing.x[:, other])


def test_temperature_grid_does_not_worsen_calibration_nll():
    logits = np.array([8.0, 4.0, -3.0, -8.0, 5.0, -4.0])
    labels = np.array([1, 0, 1, 0, 0, 1])
    fit = fit_temperature(logits, labels)
    assert fit["calibration_nll_after"] <= fit["calibration_nll_before"] + 1e-12
    assert fit["temperature"] > 1
    metrics = binary_metrics(logits, labels, fit["temperature"])
    assert 0 <= metrics["brier"] <= 1


def test_threshold_is_label_free_and_temperature_preserves_ranking():
    logits = np.linspace(-3, 3, 100)
    before = acceptance_threshold(logits, target_coverage=0.8)
    after = acceptance_threshold(logits, temperature=2, target_coverage=0.8)
    from physiobench.posttraining import sigmoid
    np.testing.assert_array_equal(sigmoid(logits) >= before, sigmoid(logits / 2) >= after)


def test_quality_event_and_curve_are_explicit():
    target = np.zeros((3, 2))
    predictions = np.array([[4.9, 2.9], [5.1, 1], [1, 3.1]])
    np.testing.assert_array_equal(correctness(predictions, target), [1, 0, 0])
    curve = risk_coverage_curve(np.array([3, 1, 0]), predictions, target)
    assert curve["failure_rate"][-1] == pytest.approx(2 / 3)
    assert curve["aurc"] == pytest.approx((0 + 0.5 + 2 / 3) / 3)


def test_risk_coverage_ties_do_not_encode_case_order():
    target = np.zeros((4, 2))
    predictions = np.array([[0, 0], [0, 0], [10, 10], [10, 10]])
    one = risk_coverage_curve(np.zeros(4), predictions, target)
    two = risk_coverage_curve(np.zeros(4), predictions[::-1], target)
    assert one["aurc"] == two["aurc"] == pytest.approx(0.5)


def test_detection_metrics_handle_ties_and_single_class():
    metrics = binary_metrics(np.zeros(4), np.array([0, 0, 1, 1]))
    assert metrics["auroc"] == pytest.approx(0.5)
    assert metrics["average_precision"] == pytest.approx(0.5)
    assert metrics["prevalence"] == pytest.approx(0.5)
    empty_auc = binary_metrics(np.zeros(4), np.zeros(4))
    assert empty_auc["auroc"] is None
    assert empty_auc["average_precision"] is None


def test_subject_conformal_uses_cluster_max_and_finite_sample_rank():
    subjects = np.repeat(np.arange(18), 2)
    target = np.zeros((36, 2))
    predictions = np.column_stack([np.repeat(np.arange(1, 19), 2), np.ones(36)])
    fit = fit_subject_conformal(predictions, target, subjects, levels=(0.5, 0.9, 0.95))
    assert fit["levels"]["0.9"]["order_statistic"] == 18
    assert fit["levels"]["0.9"]["half_width_bpm"] == [18, 1]
    assert fit["levels"]["0.95"]["half_width_bpm"] is None
