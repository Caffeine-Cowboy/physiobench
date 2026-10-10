"""Frozen-model, aggregate-only BIDMC domain-shift check.

Raw records and patient-derived predictions are never written into the report.
References are monitor HR and separately identified manual breath annotators.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import resample_poly

from .data import PhysioDataset, generate_dataset, split_subjects
from .baselines import SignalBaseline
from .metrics import cluster_bootstrap_mean

SOURCE = "https://physionet.org/files/bidmc/1.0.0/"
STARTS = (60, 180, 300)
MODEL_SEEDS = (17, 29, 43)
MODEL_ARCHITECTURES = ("cnn", "transformer")
MODEL_STAGES = ("base", "continued_base", "robust_adaptation", "selective_adaptation")


def channel_index(names: list[str], requested: str) -> int:
    """BIDMC WFDB headers include trailing commas; reject ambiguous channels."""
    canonical = [name.strip().rstrip(",").strip() for name in names]
    indices = [i for i, name in enumerate(canonical) if name == requested]
    if len(indices) != 1:
        raise ValueError(f"Expected exactly one {requested} channel")
    return indices[0]


def breath_rate(samples: np.ndarray, fs: float, start_s: float, duration_s: float = 20.) -> float:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or not np.isfinite(samples).all() or (samples < 0).any():
        raise ValueError("Breath sample positions must be finite nonnegative 1D values")
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(start_s) or start_s < 0 or not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Sampling rate and window bounds must be valid")
    times = np.sort(np.unique(samples)) / fs
    times = times[(times >= start_s) & (times < start_s + duration_s)]
    if len(times) < 2:
        return float("nan")
    intervals = np.diff(times)
    return float(60. / np.median(intervals))


def acquire(directory: Path) -> None:
    """Retrieve open licensed records into an explicitly local-only directory."""
    directory.mkdir(parents=True, exist_ok=True)
    names = [f"bidmc{i:02d}{suffix}" for i in range(1, 54)
             for suffix in (".hea", ".dat", ".breath", "n.hea", "n.dat")]
    def fetch(name: str) -> None:
        path = directory / name
        if path.exists() and path.stat().st_size > 0:
            return
        with urlopen(SOURCE + name, timeout=60) as response:
            payload = response.read(2_000_001)
        if len(payload) > 2_000_000:
            raise ValueError("Unexpected acquisition size")
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(payload)
        temporary.replace(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, names))


def load_windows(directory: Path) -> tuple[PhysioDataset, dict[str, np.ndarray], dict]:
    import wfdb  # optional external-validation dependency
    waveforms, hr, rr1, rr2, groups, cases = [], [], [], [], [], []
    excluded: Counter = Counter()
    acquisition = []
    for index in range(1, 54):
        name = f"bidmc{index:02d}"
        for suffix in (".hea", ".dat", ".breath", "n.hea", "n.dat"):
            acquisition.append(hashlib.sha256((directory / (name + suffix)).read_bytes()).hexdigest())
        record = wfdb.rdrecord(str(directory / name))
        numeric = wfdb.rdrecord(str(directory / (name + "n")))
        annotations = wfdb.rdann(str(directory / name), "breath")
        if record.fs != 125 or numeric.fs != 1:
            raise ValueError("Dataset version/sample rate differs from frozen protocol")
        channel_indices = [channel_index(record.sig_name, channel) for channel in ("II", "PLETH", "RESP")]
        hr_index = channel_index(numeric.sig_name, "HR")
        annotator_samples = {a: annotations.sample[np.array(annotations.aux_note) == a] for a in ("ann1", "ann2")}
        if set(annotations.aux_note) - {"ann1", "ann2"}:
            raise ValueError("Unknown breath annotation identity; do not silently pool")
        for start in STARTS:
            signals = record.p_signal[start * 125:(start + 20) * 125, channel_indices]
            values = numeric.p_signal[start:start + 20, hr_index]
            valid = values[np.isfinite(values) & (values >= 30) & (values <= 200)]
            if signals.shape != (2500, 3) or not np.isfinite(signals).all():
                excluded["nonfinite_or_incomplete_signals"] += 1
                continue
            if len(valid) == 0:
                excluded["missing_or_invalid_monitor_hr"] += 1
                continue
            rates = [breath_rate(annotator_samples[a], annotations.fs or record.fs, start) for a in ("ann1", "ann2")]
            x = np.zeros((4, 1000), dtype=np.float32)
            x[:3] = resample_poly(signals.T, 2, 5, axis=-1).astype(np.float32)
            waveforms.append(x)
            hr.append(float(np.median(valid)))
            rr1.append(rates[0]); rr2.append(rates[1])
            groups.append(index); cases.append(f"local-{index:02d}-{start}")
    if not waveforms:
        raise ValueError("No eligible signal windows")
    n = len(waveforms)
    targets = np.column_stack([hr, np.nan_to_num(rr1, nan=18.)]).astype(np.float32)
    dataset = PhysioDataset(x=np.stack(waveforms), observed=np.tile([True, True, True, False], (n, 1)),
        y=targets, corruption=np.zeros((n, 4), bool), severity=np.zeros((n, 4), np.float32),
        subject_id=np.asarray(groups), case_id=np.asarray(cases), scenario=np.full(n, "external"),
        condition=np.full(n, "natural"), fs=50.)
    dataset.validate()
    reference = {"hr_monitor": np.asarray(hr), "rr_ann1": np.asarray(rr1), "rr_ann2": np.asarray(rr2)}
    provenance = {"attempted_windows": 159, "included_signal_windows": n,
        "included_recordings": len(np.unique(groups)), "exclusion_counts": dict(excluded),
        "missing_rr_reference_windows": {a: int(np.isnan(reference[a]).sum()) for a in ("rr_ann1", "rr_ann2")},
        "acquisition_sha256": hashlib.sha256("\n".join(acquisition).encode()).hexdigest(),
        "acquisition_files": len(acquisition), "raw_data_distributed": False}
    return dataset, reference, provenance


def aggregate_errors(prediction: np.ndarray, dataset: PhysioDataset, reference: dict) -> dict:
    prediction = np.asarray(prediction)
    if prediction.shape != (len(dataset), 2) or not np.isfinite(prediction).all():
        raise ValueError("Predictions must be finite [N,2]; do not selectively drop failed model outputs")
    result = {}
    for name, target in reference.items():
        if name not in {"hr_monitor", "rr_ann1", "rr_ann2"}:
            raise ValueError("Unknown external reference")
        target = np.asarray(target, dtype=float)
        if target.shape != (len(dataset),):
            raise ValueError("Reference must align with dataset windows")
        channel = 0 if name == "hr_monitor" else 1
        valid = np.isfinite(target)
        errors = np.abs(prediction[valid, channel] - target[valid])
        groups = dataset.subject_id[valid]
        count = len(np.unique(groups))
        if count < 2:
            summary = {"estimate": float(errors.mean()) if count else None,
                       "ci95_low": None, "ci95_high": None, "n_subjects": count,
                       "n_bootstrap": 0, "undefined_reason": "fewer than two eligible recording clusters"}
        else:
            summary = cluster_bootstrap_mean(errors, groups, n_bootstrap=2000, seed=20261014)
        result[name] = {"eligible_windows": int(valid.sum()),
                        "units": "beats/minute" if channel == 0 else "breaths/minute",
                        **summary}
    return result


def validate_checkpoints(checkpoint_directory: Path) -> tuple[list[Path], dict[str, str]]:
    """Verify every prespecified model identity before reading external references."""
    import torch
    expected = {f"{a}-{stage}-{seed}" for a in MODEL_ARCHITECTURES
                for stage in MODEL_STAGES for seed in MODEL_SEEDS}
    paths = sorted(checkpoint_directory.glob("*.pt"))
    if {p.stem for p in paths} != expected:
        raise ValueError("Expected the exact 24 prespecified architecture/stage/seed checkpoints")
    hashes = {}
    for path in paths:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (path.stem != f"{value['architecture']}-{value['stage']}-{value['seed']}"
                or value["fs"] != 50. or value["n_samples"] != 1000):
            raise ValueError("Checkpoint metadata differs from frozen external protocol")
        hashes[path.stem] = hashlib.sha256(path.read_bytes()).hexdigest()
    return paths, hashes


def source_hash() -> str:
    h = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def run(directory: Path, checkpoint_directory: Path, report: Path) -> dict:
    import torch
    from .training import PhysioModel, predict
    torch.set_num_threads(2)
    protocol_path = report.with_suffix(".protocol.json")
    if report.exists() or protocol_path.exists():
        raise FileExistsError("Choose a new report path; refuse to overwrite frozen external evidence")
    checkpoints, hashes = validate_checkpoints(checkpoint_directory)
    frozen = {"protocol": "exploratory-frozen-BIDMC-v1", "source_code_sha256": source_hash(),
              "checkpoint_sha256": hashes, "window_starts_seconds": list(STARTS),
              "window_duration_seconds": 20, "resampling": "125 Hz to 50 Hz polyphase 2/5",
              "hr_reference": "median finite monitor HR in [30,200]; >=1 valid sample",
              "rr_reference": "separate ann1/ann2 median inter-breath rates; >=2 breaths in half-open window",
              "point_estimator": "equal-recording MAE; no external fitting or model selection",
              "bootstrap_draws": 2000, "bootstrap_seed": 20261014}
    report.parent.mkdir(parents=True, exist_ok=True)
    with protocol_path.open("x") as handle:
        handle.write(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    frozen_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    dataset, references, provenance = load_windows(directory)
    synthetic_training = split_subjects(generate_dataset())["train"]
    baseline = SignalBaseline().fit(synthetic_training).predict(dataset)["mean"]
    models = {"signal_processing": {"mae": aggregate_errors(baseline, dataset, references)}}
    for checkpoint_path in checkpoints:
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != hashes[checkpoint_path.stem]:
            raise ValueError("Checkpoint changed after external protocol freeze")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = PhysioModel(checkpoint["architecture"], checkpoint["fs"], checkpoint["n_samples"])
        model.load_state_dict(checkpoint["state_dict"])
        output = predict(model, dataset)
        models[checkpoint_path.stem] = {
            "checkpoint_sha256": hashes[checkpoint_path.stem],
            "mae": aggregate_errors(output["prediction"], dataset, references)}
    result = {"protocol": "exploratory-frozen-BIDMC-v1", "dataset": "BIDMC PPG and Respiration Dataset v1.0.0",
        "dataset_url": "https://physionet.org/content/bidmc/1.0.0/", "dataset_doi": "10.13026/C2208R",
        "dataset_attribution": "Pimentel et al.; PhysioNet BIDMC PPG and Respiration Dataset",
        "dataset_license": "Open Data Commons Attribution License v1.0", "provenance": provenance,
        "frozen_protocol_sha256": frozen_hash, "source_code_sha256": frozen["source_code_sha256"],
        "point_estimator": "equal-recording mean absolute error", "bootstrap_unit": "source recording",
        "bootstrap_draws": 2000, "models": models,
        "limitations": ["Exploratory domain-shift check; no clinical efficacy or diagnostic claim",
          "HR reference is monitor-derived; RR references are separate manual annotators",
          "Three prespecified windows per recording; temporal stationarity and ICU selection limit transportability",
          "No external fitting, model selection, threshold tuning or recalibration",
          "Natural corruption lacks ground-truth annotation; detector metrics not reported",
          "Recordings are bootstrap clusters; independent patient identity cannot be established from released records",
          "Pointwise intervals are not adjusted for multiple models or outcomes"]}
    if source_hash() != frozen["source_code_sha256"] or hashlib.sha256(protocol_path.read_bytes()).hexdigest() != frozen_hash:
        raise ValueError("Evaluation source/protocol changed after freeze; refuse publication")
    with report.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-data", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        acquire(args.local_data)
    result = run(args.local_data, args.checkpoints, args.report)
    print(json.dumps({"provenance": result["provenance"], "models_evaluated": len(result["models"])}))


if __name__ == "__main__":
    main()
