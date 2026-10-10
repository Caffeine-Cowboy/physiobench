"""Reproducible train / adapt / calibrate / test experiment runner.

Run: python -m physiobench.experiments --config configs/experiment-study.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

from .data import (PhysioDataset, generate_dataset, split_subjects, apply_stress,
                   concatenate, STRESS_CONDITIONS)
from .training import PhysioModel, TrainConfig, predict, seed_everything, train
from .posttraining import (fit_temperature, binary_metrics, correctness, sigmoid,
                           acceptance_threshold, risk_coverage_curve, fit_subject_conformal)


STAGES = ("base", "continued_base", "robust_adaptation", "selective_adaptation")
MODALITIES = ("ecg", "ppg", "respiration", "motion")
TEST_CONDITIONS = tuple(STRESS_CONDITIONS) + tuple(f"drop_{name}" for name in MODALITIES)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest_arrays(dataset: PhysioDataset) -> str:
    checksum = hashlib.sha256()
    for name in ("x", "observed", "y", "corruption", "severity", "subject_id", "case_id", "scenario"):
        array = np.ascontiguousarray(getattr(dataset, name))
        checksum.update(name.encode())
        checksum.update(str(array.shape).encode())
        checksum.update(str(array.dtype).encode())
        checksum.update(array.tobytes())
    return checksum.hexdigest()


def split_validation(dataset: PhysioDataset, seed: int) -> tuple[PhysioDataset, PhysioDataset]:
    subjects = np.random.default_rng(seed).permutation(np.unique(dataset.subject_id))
    midpoint = len(subjects) // 2
    if midpoint < 2:
        raise ValueError("Need >=4 validation subjects for disjoint tuning/calibration")
    tuning = dataset.subset(np.isin(dataset.subject_id, subjects[:midpoint]))
    calibration = dataset.subset(np.isin(dataset.subject_id, subjects[midpoint:]))
    assert not set(tuning.subject_id) & set(calibration.subject_id)
    return tuning, calibration


def stress(dataset: PhysioDataset, condition: str, seed: int) -> PhysioDataset:
    if condition.startswith("drop_"):
        name = condition[5:]
        if name not in MODALITIES:
            raise ValueError(f"Unknown missing-modality ablation {name}")
        channel = MODALITIES.index(name)
        result = dataset.subset(np.arange(len(dataset)))
        result.x[:, channel] = 0
        result.observed[:, channel] = False
        result.corruption[:, channel] = True
        result.severity[:, channel] = 1
        result.condition[:] = condition
        result.validate()
        return result
    return apply_stress(dataset, condition, seed=seed)


def mixture(dataset: PhysioDataset, conditions: tuple[str, ...], seed: int) -> PhysioDataset:
    return concatenate([stress(dataset, condition, seed) for condition in conditions])


def metrics(output: dict[str, np.ndarray], dataset: PhysioDataset,
            calibration: dict[str, Any]) -> dict[str, Any]:
    error = np.abs(output["prediction"] - dataset.y)
    labels = correctness(output["prediction"], dataset.y)
    q_temperature = calibration["quality"]["temperature"]
    threshold = calibration["acceptance_threshold"]
    accepted = sigmoid(output["quality_logits"] / q_temperature) >= threshold
    accepted_count = int(accepted.sum())
    rc = risk_coverage_curve(output["quality_logits"], output["prediction"], dataset.y, q_temperature)
    result = {
        "n_cases": len(dataset), "n_subjects": len(np.unique(dataset.subject_id)),
        "hr_mae_bpm": float(error[:, 0].mean()), "rr_mae_bpm": float(error[:, 1].mean()),
        "hr_rmse_bpm": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rr_rmse_bpm": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "failure_rate": float(1 - labels.mean()), "coverage": float(accepted.mean()),
        "accepted_cases": accepted_count,
        "selective_hr_mae_bpm": float(error[accepted, 0].mean()) if accepted_count else None,
        "selective_rr_mae_bpm": float(error[accepted, 1].mean()) if accepted_count else None,
        "selective_failure_rate": float(1 - labels[accepted].mean()) if accepted_count else None,
        "quality_uncalibrated": binary_metrics(output["quality_logits"], labels),
        "quality_calibrated": binary_metrics(output["quality_logits"], labels, q_temperature),
        "corruption_uncalibrated": binary_metrics(output["corruption_logits"], dataset.corruption,
                                                   mask=dataset.observed),
        "corruption_calibrated": binary_metrics(output["corruption_logits"], dataset.corruption,
                                                 calibration["corruption"]["temperature"], mask=dataset.observed),
        "available_sensor_severity_mae": float(np.abs(output["severity_prediction"] - dataset.severity)[dataset.observed].mean()),
        "risk_coverage": rc,
    }
    # Always-observed motion is a true-negative control in this simulator, so
    # report modality Brier instead of presenting undefined single-class AUROC.
    result["modality_corruption_brier"] = {
        name: binary_metrics(output["corruption_logits"][:, i], dataset.corruption[:, i],
                             calibration["corruption"]["temperature"], mask=dataset.observed[:, i])["brier"]
        if np.any(dataset.observed[:, i]) else None for i, name in enumerate(MODALITIES)
    }
    intervals = {}
    for level, specification in calibration["regression_intervals"]["levels"].items():
        width = specification["half_width_bpm"]
        if width is None:
            intervals[level] = None
            continue
        covered = error <= np.asarray(width)
        subject_coverage = np.stack([covered[dataset.subject_id == subject].all(axis=0)
                                     for subject in np.unique(dataset.subject_id)]).mean(axis=0)
        intervals[level] = {"case_coverage_hr": float(covered[:, 0].mean()),
                            "case_coverage_rr": float(covered[:, 1].mean()),
                            "subject_coverage_hr": float(subject_coverage[0]),
                            "subject_coverage_rr": float(subject_coverage[1]),
                            "width_hr_bpm": float(2 * width[0]), "width_rr_bpm": float(2 * width[1])}
    result["regression_intervals"] = intervals
    return result


def calibrate(model: PhysioModel, calibration_data: PhysioDataset, coverage: float) -> dict[str, Any]:
    output = predict(model, calibration_data)
    q_fit = fit_temperature(output["quality_logits"], correctness(output["prediction"], calibration_data.y))
    c_fit = fit_temperature(output["corruption_logits"], calibration_data.corruption, mask=calibration_data.observed)
    return {"quality": q_fit, "corruption": c_fit,
            "acceptance_threshold": acceptance_threshold(output["quality_logits"], q_fit["temperature"], coverage),
            "target_coverage": coverage, "n_subjects": len(np.unique(calibration_data.subject_id)),
            "regression_intervals": fit_subject_conformal(output["prediction"], calibration_data.y,
                                                           calibration_data.subject_id),
            "subject_ids": np.unique(calibration_data.subject_id).tolist()}


def paired_subject_bootstrap(prediction_files: dict[tuple[str, str, int, str, str], Path],
                            seeds: list[int], draws: int, seed: int) -> list[dict[str, Any]]:
    """Bootstrap subjects after averaging errors over the fixed training seeds.

    No iid-window CI: all windows/stresses for one subject remain together.
    Intervals are conditional on the published simulator and these fixed seeds;
    they do not capture distributional or clinical uncertainty.
    """
    rng = np.random.default_rng(seed)
    rows = []
    comparisons = (("robust_adaptation", "continued_base"), ("selective_adaptation", "robust_adaptation"),
                   ("selective_adaptation", "base"))
    architectures = sorted({key[0] for key in prediction_files})
    for architecture in architectures:
        for split in ("test_id", "test_ood"):
            for condition in TEST_CONDITIONS:
                for treatment, control in comparisons:
                    differences = []
                    subject_ids = None
                    for model_seed in seeds:
                        with np.load(prediction_files[(architecture, treatment, model_seed, split, condition)]) as a, \
                             np.load(prediction_files[(architecture, control, model_seed, split, condition)]) as b:
                            if not np.array_equal(a["case_id"], b["case_id"]) or not np.array_equal(a["y"], b["y"]) or \
                               not np.array_equal(a["subject_id"], b["subject_id"]):
                                raise ValueError("Nonpaired predictions in bootstrap")
                            ids = np.unique(a["subject_id"])
                            if subject_ids is not None and not np.array_equal(ids, subject_ids):
                                raise ValueError("Different test subjects across model seeds")
                            subject_ids = ids
                            delta = np.abs(a["prediction"] - a["y"]) - np.abs(b["prediction"] - b["y"])
                            differences.append(np.stack([delta[a["subject_id"] == subject].mean(axis=0) for subject in ids]))
                    averaged = np.stack(differences).mean(axis=0)
                    indices = rng.integers(0, len(subject_ids), size=(draws, len(subject_ids)))
                    boot = averaged[indices].mean(axis=1)
                    for dimension, metric in enumerate(("hr_mae_bpm", "rr_mae_bpm")):
                        low, high = np.quantile(boot[:, dimension], [0.025, 0.975])
                        rows.append({"architecture": architecture, "split": split, "condition": condition,
                                     "treatment": treatment, "control": control, "metric": metric,
                                     "delta": float(averaged[:, dimension].mean()), "ci95": [float(low), float(high)],
                                     "n_subjects": len(subject_ids), "bootstrap_draws": draws,
                                     "interpretation": "negative delta favors treatment; pointwise, not multiplicity-adjusted"})
    return rows


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["architecture"], row["stage"], row["split"], row["condition"])
        groups.setdefault(key, []).append(row["metrics"])
    numeric = ("hr_mae_bpm", "rr_mae_bpm", "failure_rate", "coverage", "selective_failure_rate",
               "selective_hr_mae_bpm", "selective_rr_mae_bpm", "available_sensor_severity_mae")
    result = []
    for (architecture, stage, split, condition), values in groups.items():
        aggregate = {"architecture": architecture, "stage": stage, "split": split, "condition": condition,
                     "n_seeds": len(values), "n_cases_per_seed": values[0]["n_cases"],
                     "n_subjects": values[0]["n_subjects"]}
        for key in numeric:
            observed_values = [entry[key] for entry in values if entry[key] is not None]
            aggregate[key] = ({"mean": float(np.mean(observed_values)),
                               "seed_sd": float(np.std(observed_values, ddof=1)) if len(observed_values) > 1 else 0.0}
                              if observed_values else None)
        for kind in ("quality_uncalibrated", "quality_calibrated", "corruption_uncalibrated", "corruption_calibrated"):
            aggregate[kind] = {key: {"mean": float(np.mean([entry[kind][key] for entry in values])),
                                     "seed_sd": float(np.std([entry[kind][key] for entry in values], ddof=1)) if len(values) > 1 else 0.0}
                               for key in ("nll", "brier", "ece")}
        aggregate["aurc"] = {"mean": float(np.mean([entry["risk_coverage"]["aurc"] for entry in values])),
                              "seed_sd": float(np.std([entry["risk_coverage"]["aurc"] for entry in values], ddof=1)) if len(values) > 1 else 0.0}
        for kind in ("corruption_uncalibrated", "corruption_calibrated"):
            for metric in ("auroc", "average_precision", "prevalence"):
                valid = [entry[kind][metric] for entry in values if entry[kind][metric] is not None]
                aggregate[kind][metric] = {"mean": float(np.mean(valid)),
                                           "seed_sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0} if valid else None
        aggregate["regression_intervals"] = {}
        for level in values[0]["regression_intervals"]:
            if values[0]["regression_intervals"][level] is None:
                aggregate["regression_intervals"][level] = None
                continue
            aggregate["regression_intervals"][level] = {
                metric: {"mean": float(np.mean([entry["regression_intervals"][level][metric] for entry in values])),
                         "seed_sd": float(np.std([entry["regression_intervals"][level][metric] for entry in values], ddof=1)) if len(values) > 1 else 0.0}
                for metric in values[0]["regression_intervals"][level]}
        result.append(aggregate)
    return result


def run_study(configuration: dict[str, Any], output_directory: Path) -> dict[str, Any]:
    started = time.time()
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "configuration.json", configuration)
    dataset = generate_dataset(**configuration["dataset"])
    splits = split_subjects(dataset, seed=configuration["split_seed"])
    tuning, calibration = split_validation(splits["val"], configuration["calibration_split_seed"])
    stress_seed = configuration["stress_seed"]
    base_train = mixture(splits["train"], ("clean", "motion_artifact"), stress_seed)
    robust_train = mixture(splits["train"], tuple(STRESS_CONDITIONS), stress_seed)
    # Both post-training controls have exactly the same case/optimizer-step budget.
    continued_train = mixture(splits["train"], ("clean", "motion_artifact", "clean", "motion_artifact", "clean"), stress_seed)
    tuning_mix = mixture(tuning, tuple(STRESS_CONDITIONS), stress_seed)
    calibration_mix = mixture(calibration, tuple(STRESS_CONDITIONS), stress_seed)
    test_data = {(name, condition): stress(splits[name], condition, stress_seed)
                 for name in ("test_id", "test_ood") for condition in TEST_CONDITIONS}
    base_config = TrainConfig(**configuration["training"])
    post_config = replace(base_config, **configuration["posttraining"])
    rows, fits, prediction_files = [], [], {}
    for architecture in configuration["architectures"]:
        for model_seed in configuration["seeds"]:
            seed_everything(model_seed, base_config.cpu_threads)
            base = PhysioModel(architecture, dataset.fs, dataset.x.shape[-1])
            print(f"Training {architecture}, seed={model_seed}, base", flush=True)
            fit_started = time.time()
            base_fit = train(base, base_train, tuning_mix, base_config, model_seed)
            base_state = {key: value.clone() for key, value in base.state_dict().items()}
            base_fit.update({"architecture": architecture, "seed": model_seed, "stage": "base",
                             "duration_s": time.time() - fit_started, "train_cases": len(base_train)})
            fits.append(base_fit)
            for stage in STAGES:
                model = PhysioModel(architecture, dataset.fs, dataset.x.shape[-1])
                model.load_state_dict(base_state)
                if stage != "base":
                    adaptation = continued_train if stage == "continued_base" else robust_train
                    print(f"Training {architecture}, seed={model_seed}, {stage}", flush=True)
                    fit_started = time.time()
                    report = train(model, adaptation, tuning_mix, post_config, model_seed + 10000,
                                   selective=stage == "selective_adaptation")
                    report.update({"architecture": architecture, "seed": model_seed, "stage": stage,
                                   "duration_s": time.time() - fit_started, "train_cases": len(adaptation)})
                    fits.append(report)
                cal = calibrate(model, calibration_mix, base_config.target_coverage)
                checkpoint_path = output_directory / "checkpoints" / f"{architecture}-{stage}-{model_seed}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "architecture": architecture,
                            "fs": dataset.fs, "n_samples": dataset.x.shape[-1], "seed": model_seed,
                            "stage": stage, "calibration": cal}, checkpoint_path)
                for (split, condition), cases in test_data.items():
                    output = predict(model, cases)
                    result = metrics(output, cases, cal)
                    rows.append({"architecture": architecture, "seed": model_seed, "stage": stage,
                                 "split": split, "condition": condition, "metrics": result, "calibration": cal})
                    path = output_directory / "predictions" / f"{architecture}-{stage}-{model_seed}-{split}-{condition}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, **output, y=cases.y, observed=cases.observed,
                                        corruption=cases.corruption, subject_id=cases.subject_id,
                                        case_id=cases.case_id, scenario=cases.scenario)
                    prediction_files[(architecture, stage, model_seed, split, condition)] = path
                # Incremental persistence: an interrupted long run is auditable.
                write_json(output_directory / "per_run_metrics.json", rows)
                write_json(output_directory / "training_history.json", fits)
    aggregate = aggregate_metrics(rows)
    bootstrap = paired_subject_bootstrap(prediction_files, configuration["seeds"],
                                         configuration["bootstrap_draws"], configuration["bootstrap_seed"])
    write_json(output_directory / "aggregate_metrics.json", aggregate)
    write_json(output_directory / "paired_subject_bootstrap.json", bootstrap)
    source = Path(__file__).parent
    manifest = {"protocol_version": 1, "status": "completed", "configuration": configuration,
                "data_sha256": digest_arrays(dataset),
                "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source.glob("*.py"))},
                "split_subjects": {**{name: np.unique(value.subject_id).tolist() for name, value in splits.items()},
                                   "tuning": np.unique(tuning.subject_id).tolist(),
                                   "calibration": np.unique(calibration.subject_id).tolist()},
                "split_cases": {name: len(value) for name, value in splits.items()},
                "n_model_fits": len(fits), "n_evaluation_rows": len(rows),
                "duration_s": time.time() - started, "hardware": {"platform": platform.platform(),
                    "machine": platform.machine(), "processor": platform.processor(), "device": "cpu",
                    "cpu_threads": base_config.cpu_threads},
                "software": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
                "limitations": ["Entire primary study uses synthetic data; no clinical efficacy claim",
                                 "OOD means held-out equation families within one simulator",
                                 "Fixed development distribution and three training seeds",
                                 "Pointwise subject-bootstrap CIs; no multiplicity adjustment",
                                 "Post-training means supervised signal-model adaptation, not LLM SFT or RL"]}
    manifest["artifacts_sha256"] = {str(path.relative_to(output_directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sorted(output_directory.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    write_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment-study.json"))
    parser.add_argument("--output", type=Path, default=Path("results/neural-study"))
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text())
    manifest = run_study(configuration, arguments.output)
    print(json.dumps({"status": manifest["status"], "duration_s": manifest["duration_s"],
                      "n_model_fits": manifest["n_model_fits"], "results": str(arguments.output)}), flush=True)


if __name__ == "__main__":
    main()
