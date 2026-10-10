import numpy as np
import pytest
from physiobench.data import generate_dataset
from physiobench.external_validation import (aggregate_errors, breath_rate, channel_index,
                                           MODEL_ARCHITECTURES, MODEL_SEEDS, MODEL_STAGES,
                                           validate_checkpoints)


def test_manual_reference_uses_annotation_identity_upstream_and_window_boundaries():
    # Boundaries are half-open and duplicate events cannot increase a rate.
    samples = np.array([59, 60, 64, 64, 68, 72, 76, 80]) * 125
    assert breath_rate(samples, 125., 60.) == 15.
    assert np.isnan(breath_rate(np.array([60]) * 125, 125., 60.))


def test_manual_reference_respects_annotation_sampling_rate():
    assert breath_rate(np.array([0, 500, 1000]), 125., 0.) == 15.


def test_wfdb_channel_punctuation_and_ambiguity():
    assert channel_index(["RESP,", "PLETH,", "V,", "AVR,", "II,"], "II") == 4
    assert channel_index(["HR,", "PULSE,"], "HR") == 0
    with pytest.raises(ValueError, match="exactly one"):
        channel_index(["HR", "HR,"], "HR")
    with pytest.raises(ValueError, match="exactly one"):
        channel_index(["HR,"], "II")


def test_invalid_annotation_rate_and_positions_fail():
    with pytest.raises(ValueError):
        breath_rate(np.array([0, 500]), 0., 0.)
    with pytest.raises(ValueError):
        breath_rate(np.array([0, np.nan]), 125., 0.)


def test_external_aggregate_handles_missing_reference_and_weights_recordings_equally():
    dataset = generate_dataset(n_subjects=10).subset([0, 1, 6])
    predicted = np.array([[11., 15.], [11., 15.], [19., 15.]])
    refs = {"hr_monitor": np.array([10., 10., 10.]),
            "rr_ann1": np.full(3, np.nan), "rr_ann2": np.array([12., np.nan, np.nan])}
    result = aggregate_errors(predicted, dataset, refs)
    assert result["hr_monitor"]["estimate"] == 5.  # recording means 1 and 9, not window mean
    assert result["rr_ann1"]["estimate"] is None
    assert result["rr_ann1"]["eligible_windows"] == 0
    assert result["rr_ann2"]["estimate"] == 3.
    assert result["rr_ann2"]["ci95_low"] is None
    assert "fewer than two" in result["rr_ann2"]["undefined_reason"]
    with pytest.raises(ValueError, match="finite"):
        aggregate_errors(predicted * np.nan, dataset, refs)
    with pytest.raises(ValueError, match="align"):
        aggregate_errors(predicted, dataset, {"hr_monitor": np.array([10.])})


def test_exact_checkpoint_set_and_metadata_required_before_external_data(tmp_path):
    import torch
    for architecture in MODEL_ARCHITECTURES:
        for stage in MODEL_STAGES:
            for seed in MODEL_SEEDS:
                torch.save({"architecture": architecture, "stage": stage, "seed": seed,
                            "fs": 50., "n_samples": 1000},
                           tmp_path / f"{architecture}-{stage}-{seed}.pt")
    paths, hashes = validate_checkpoints(tmp_path)
    assert len(paths) == len(hashes) == 24
    example = paths[0]
    value = torch.load(example, weights_only=True)
    value["fs"] = 125.
    torch.save(value, example)
    with pytest.raises(ValueError, match="metadata"):
        validate_checkpoints(tmp_path)
    example.unlink()
    with pytest.raises(ValueError, match="exact 24"):
        validate_checkpoints(tmp_path)
