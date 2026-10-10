"""Scientific invariants: determinism, leakage boundaries, paired stresses, metrics."""
import numpy as np
import pytest
from physiobench.data import (generate_dataset, split_subjects, apply_stress,
                              save_dataset, load_dataset, STRESS_CONDITIONS)
from physiobench.baselines import waveform_features, SignalBaseline
from physiobench.metrics import evaluate_predictions, paired_cluster_bootstrap, cluster_bootstrap_mean, selective_risk_curve
from physiobench.simulator import KNOWN_SCENARIOS, OOD_SCENARIOS


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_subjects=20, windows_per_subject=6, seed=123)


def test_generation_deterministic_and_truth_consistent(dataset):
    repeat = generate_dataset(n_subjects=20, windows_per_subject=6, seed=123)
    np.testing.assert_array_equal(dataset.x,repeat.x)
    np.testing.assert_allclose(dataset.y[:,0],dataset.latent_hr.mean(axis=1),rtol=1e-6)
    np.testing.assert_allclose(dataset.y[:,1],dataset.latent_rr.mean(axis=1),rtol=1e-6)
    assert not dataset.corruption.any()
    assert set(dataset.scenario) == set(KNOWN_SCENARIOS+OOD_SCENARIOS)


def test_subject_and_scenario_leakage_boundaries(dataset):
    parts = split_subjects(dataset,seed=43)
    assert not set(parts['train'].subject_id)&set(parts['val'].subject_id)
    assert not set(parts['train'].subject_id)&set(parts['test_id'].subject_id)
    assert not set(parts['val'].subject_id)&set(parts['test_ood'].subject_id)
    assert set(parts['train'].scenario) == set(KNOWN_SCENARIOS)
    assert set(parts['test_ood'].scenario) == set(OOD_SCENARIOS)
    assert not set(parts['test_id'].case_id)&set(parts['test_ood'].case_id)


@pytest.mark.parametrize('condition',STRESS_CONDITIONS)
def test_stresses_preserve_truth_and_case_identity(dataset,condition):
    stress = apply_stress(dataset,condition,seed=44)
    stress.validate()
    np.testing.assert_array_equal(stress.y,dataset.y)
    np.testing.assert_array_equal(stress.case_id,dataset.case_id)
    np.testing.assert_array_equal(stress.subject_id,dataset.subject_id)
    indices = np.asarray([9,2,50,1])
    local = apply_stress(dataset.subset(indices),condition,seed=44)
    np.testing.assert_array_equal(local.x,stress.x[indices])
    if condition != "clean":
        assert stress.corruption.any()
        assert not np.array_equal(stress.x,dataset.x)
    if condition == "missing_modality":
        assert np.all(stress.observed.sum(axis=1)==3)
        assert not np.any(stress.x[~stress.observed])


def test_feature_extractor_cannot_leak_labels(dataset):
    features = waveform_features(dataset)
    altered = dataset.subset(np.arange(len(dataset)))
    altered.y[:] = 999
    altered.corruption[:] = True
    altered.severity[:] = 1
    altered.subject_id[:] = 999
    altered.scenario[:] = 'changed'
    altered.latent_hr[:] = 0
    altered.latent_rr[:] = 0
    np.testing.assert_array_equal(features,waveform_features(altered))
    assert features.shape == (len(dataset),70)
    assert np.isfinite(features).all()


def test_separate_storage_and_checksum(dataset,tmp_path):
    manifest = save_dataset(dataset,tmp_path)
    assert manifest['n_cases'] == len(dataset)
    with np.load(tmp_path/'observations.npz',allow_pickle=False) as obs:
        assert 'y' not in obs and 'latent_hr' not in obs and 'severity' not in obs
    restored = load_dataset(tmp_path)
    np.testing.assert_array_equal(restored.x,dataset.x)
    np.testing.assert_array_equal(restored.y,dataset.y)
    with (tmp_path/'labels.npz').open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(ValueError,match='Checksum'):
        load_dataset(tmp_path)


def test_metrics_perfect_prediction_and_undefined_auc(dataset):
    report = evaluate_predictions(dataset,dataset.y,np.ones_like(dataset.y),
                                  np.zeros_like(dataset.severity),np.zeros_like(dataset.severity))
    assert report['hr_mae'] == 0
    assert report['rr_mae'] == 0
    assert report['hr_coverage_0.95'] == 1
    assert report['corruption_macro_auroc'] is None
    assert report['corruption']['ecg']['auroc'] is None
    assert report['severity_mae'] == 0


def test_cluster_bootstrap_estimand_and_pairing(dataset):
    # Unequal windows: equal-subject estimand should be (0+10)/2, not 2.5.
    report = cluster_bootstrap_mean(np.array([0,0,0,10]),np.array([0,0,0,1]),200,seed=4)
    assert report['estimate'] == 5
    stress = apply_stress(dataset,'contradiction').subset(np.arange(len(dataset))[::-1])
    baseline = dataset.y+2
    candidate = stress.y+1
    paired = paired_cluster_bootstrap(dataset,stress,baseline,candidate,n_bootstrap=200)
    assert paired['estimate'] == -1
    assert paired['ci95_low'] == paired['ci95_high'] == -1
    with pytest.raises(ValueError,match='identical case'):
        paired_cluster_bootstrap(dataset,stress.subset(np.arange(2)),baseline,candidate[:2])


def test_signal_baseline_has_finite_outputs(dataset):
    parts = split_subjects(dataset)
    prediction = SignalBaseline().fit(parts['val']).predict(parts['test_id'])
    assert prediction['mean'].shape == parts['test_id'].y.shape
    report = evaluate_predictions(parts['test_id'],prediction)
    # These loose limits detect broken waveform conventions, not clinical accuracy.
    assert report['hr_mae'] < 15
    assert report['rr_mae'] < 7


def test_constant_confidence_has_no_order_dependent_selective_gain():
    y = np.zeros((4,2))
    prediction = np.array([[0,0],[0,0],[10,4],[10,4]])
    report = selective_risk_curve(y,prediction,np.ones((4,2)))
    assert all(value == 1 for value in report.values())


@pytest.mark.parametrize('damage', ['mask_dtype','missing_nonzero','target_shape','time_empty','nan_fs','latent_shape'])
def test_invalid_schema_fails(dataset,damage):
    copy = dataset.subset(np.arange(2))
    if damage=='mask_dtype': copy.observed = copy.observed.astype(int)
    if damage=='missing_nonzero': copy.observed[0,0]=False
    if damage=='target_shape': copy.y = copy.y[:,:1]
    if damage=='time_empty': copy.x = copy.x[:,:,:0]
    if damage=='nan_fs': copy.fs = float('nan')
    if damage=='latent_shape': copy.latent_hr = copy.latent_hr[:,:3]
    with pytest.raises(ValueError): copy.validate()
