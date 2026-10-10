"""Dataset contracts, subject splits, paired stress transformations, and storage."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Sequence
import numpy as np

from .simulator import (MODALITIES, KNOWN_SCENARIOS, OOD_SCENARIOS, SCENARIOS,
                        sample_subject, simulate_window, ppg_from_phase)

STRESS_CONDITIONS = ("clean", "missing_modality", "motion_artifact", "temporal_offset", "contradiction")


@dataclass
class PhysioDataset:
    x: np.ndarray
    observed: np.ndarray
    y: np.ndarray
    corruption: np.ndarray
    severity: np.ndarray
    subject_id: np.ndarray
    case_id: np.ndarray
    scenario: np.ndarray
    condition: np.ndarray
    fs: float = 50.0
    latent_hr: np.ndarray | None = None
    latent_rr: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.x)

    def subset(self, indices: np.ndarray | Sequence[int]) -> "PhysioDataset":
        indices = np.asarray(indices)
        return PhysioDataset(**{
            field.name: (value[indices].copy() if isinstance(value, np.ndarray) else value)
            for field in fields(self)
            for value in [getattr(self, field.name)]
        })

    def validate(self) -> None:
        n = len(self)
        if self.x.ndim != 3 or self.x.shape[1] != 4 or self.x.shape[2] < 2 or n == 0:
            raise ValueError("x must have shape [N,4,T]")
        if self.observed.shape != (n, 4) or self.y.shape != (n, 2):
            raise ValueError("Incorrect mask/target dimensions")
        if self.corruption.shape != (n, 4) or self.severity.shape != (n, 4):
            raise ValueError("Incorrect corruption dimensions")
        if self.observed.dtype != np.bool_ or self.corruption.dtype != np.bool_:
            raise ValueError("Observed/corruption masks must be boolean")
        if any(len(getattr(self, name)) != n for name in ("subject_id", "case_id", "scenario", "condition")):
            raise ValueError("Metadata lengths differ")
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError("Nonfinite waveform or target")
        if (self.y <= 0).any():
            raise ValueError("Rate targets must be positive")
        if not ((self.severity >= 0) & (self.severity <= 1)).all():
            raise ValueError("Severity must be in [0,1]")
        if not np.isfinite(self.fs) or self.fs <= 0:
            raise ValueError("Sample rate must be positive")
        if np.any(self.x[~self.observed] != 0):
            raise ValueError("Unobserved channels must be zero-filled")
        for latent in (self.latent_hr, self.latent_rr):
            if latent is not None and (latent.shape != (n,self.x.shape[2]) or not np.isfinite(latent).all()):
                raise ValueError("Latent trajectories must have shape [N,T] and be finite")


def generate_dataset(
    n_subjects: int = 180, windows_per_subject: int = 6,
    duration_s: float = 20.0, fs: float = 50.0, seed: int = 20261010,
) -> PhysioDataset:
    if n_subjects < 10 or windows_per_subject < 6:
        raise ValueError("Use >=10 subjects and >=6 windows/subject to cover every family")
    rng = np.random.default_rng(seed)
    rows = []
    for subject_id in range(n_subjects):
        subject = sample_subject(rng)
        # Each family appears for every subject; unknown families are later excluded
        # from development splits, not accidentally sampled into training.
        for window in range(windows_per_subject):
            scenario = SCENARIOS[window % len(SCENARIOS)]
            signals, targets, latent = simulate_window(subject, scenario, rng, fs=fs, duration_s=duration_s)
            rows.append((signals, targets, subject_id, f"s{subject_id:04d}-w{window:03d}", scenario, latent))
    n = len(rows)
    dataset = PhysioDataset(
        x=np.stack([row[0] for row in rows]), observed=np.ones((n, 4), bool),
        y=np.stack([row[1] for row in rows]), corruption=np.zeros((n, 4), bool),
        severity=np.zeros((n, 4), np.float32),
        subject_id=np.asarray([row[2] for row in rows], np.int64),
        case_id=np.asarray([row[3] for row in rows]), scenario=np.asarray([row[4] for row in rows]),
        condition=np.full(n, "clean", dtype="U24"), fs=fs,
        latent_hr=np.stack([row[5]["hr_bpm"] for row in rows]),
        latent_rr=np.stack([row[5]["rr_bpm"] for row in rows]),
    )
    dataset.validate()
    return dataset


def split_subjects(
    dataset: PhysioDataset, seed: int = 20261010,
    train_fraction: float = 0.6, val_fraction: float = 0.2,
) -> dict[str, PhysioDataset]:
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("Train/validation fractions must be positive and sum to <1")
    subjects = np.random.default_rng(seed).permutation(np.unique(dataset.subject_id))
    a = int(len(subjects) * train_fraction)
    b = a + int(len(subjects) * val_fraction)
    if a < 1 or b <= a or b >= len(subjects):
        raise ValueError("Insufficient subjects for requested split")
    groups = {"train": subjects[:a], "val": subjects[a:b], "test_id": subjects[b:], "test_ood": subjects[b:]}
    result = {}
    for name, ids in groups.items():
        families = OOD_SCENARIOS if name == "test_ood" else KNOWN_SCENARIOS
        mask = np.isin(dataset.subject_id, ids) & np.isin(dataset.scenario, families)
        result[name] = dataset.subset(mask)
    assert_split_integrity(result)
    return result


def assert_split_integrity(splits: dict[str, PhysioDataset]) -> None:
    train = set(splits["train"].subject_id)
    val = set(splits["val"].subject_id)
    test = set(splits["test_id"].subject_id) | set(splits["test_ood"].subject_id)
    if train & val or train & test or val & test:
        raise ValueError("Subject leakage between development and test")
    for name in ("train", "val", "test_id"):
        if set(splits[name].scenario) - set(KNOWN_SCENARIOS):
            raise ValueError("Held-out scenario appeared in development/ID split")
    if set(splits["test_ood"].scenario) - set(OOD_SCENARIOS):
        raise ValueError("Known scenario appeared in OOD split")


def _case_rng(seed: int, case_id: str, condition: str) -> np.random.Generator:
    key = f"{seed}:{case_id}:{condition}".encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return np.random.default_rng(value)


def apply_stress(dataset: PhysioDataset, condition: str, seed: int = 20261011,
                 intensity: float = 1.0) -> PhysioDataset:
    """Transform observations of identical cases; physiological labels stay fixed.

    Per-case RNG makes transformations invariant to order/subsetting. Severity is
    a simulator control label, not clinical severity or estimated disease burden.
    """
    if condition not in STRESS_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if not 0 < intensity <= 1:
        raise ValueError("intensity must be in (0,1]")
    result = dataset.subset(np.arange(len(dataset)))
    result.condition[:] = condition
    if condition == "clean":
        return result
    t = np.arange(result.x.shape[2]) / result.fs
    for i, case in enumerate(result.case_id):
        rng = _case_rng(seed, str(case), condition)
        if condition == "missing_modality":
            channel = int(rng.integers(0, 3))
            result.x[i, channel] = 0
            result.observed[i, channel] = False
            result.corruption[i, channel] = True
            result.severity[i, channel] = 1
        elif condition == "motion_artifact":
            artifact = (np.sin(2*np.pi*rng.uniform(0.7, 2.8)*t + rng.uniform(0, 6))
                        + 0.5 * rng.normal(size=t.size))
            amplitude = rng.uniform(0.3, 1.0) * intensity
            result.x[i, 3] += amplitude * artifact
            for channel, coupling in ((0, 0.45), (1, 0.8)):
                result.x[i, channel] += amplitude * coupling * artifact
                result.corruption[i, channel] = True
                result.severity[i, channel] = amplitude
        elif condition == "temporal_offset":
            shift_s = rng.uniform(0.8, 2.4) * intensity
            result.x[i, 1] = np.roll(result.x[i, 1], int(round(shift_s * result.fs)))
            result.corruption[i, 1] = True
            result.severity[i, 1] = shift_s / 2.4
        elif condition == "contradiction":
            # An inconsistent peripheral pulse while electrical HR is unchanged.
            # Stress construction may use label truth; models/feature extraction may not.
            relative_error = rng.uniform(0.20, 0.45) * intensity
            direction = rng.choice([-1, 1])
            decoy_hr = float(np.clip(result.y[i, 0] * (1 + direction * relative_error), 35, 180))
            phase = rng.uniform(0, 1) + np.cumsum(np.full(t.size, decoy_hr / (60*result.fs)))
            gain = max(float(result.x[i, 1].std()), 0.05) / 0.29
            result.x[i, 1] = gain * ppg_from_phase(phase) + rng.normal(0, 0.03, t.size)
            result.corruption[i, 1] = True
            result.severity[i, 1] = relative_error / 0.45
    result.validate()
    return result


def concatenate(datasets: Sequence[PhysioDataset]) -> PhysioDataset:
    if not datasets or len({d.fs for d in datasets}) != 1:
        raise ValueError("Provide datasets with a shared sample rate")
    return PhysioDataset(**{
        field.name: (None if getattr(datasets[0], field.name) is None else
                     np.concatenate([getattr(d, field.name) for d in datasets])
                     if isinstance(getattr(datasets[0], field.name), np.ndarray) else
                     getattr(datasets[0], field.name))
        for field in fields(PhysioDataset)
    })


def save_dataset(dataset: PhysioDataset, directory: str | Path) -> dict:
    """Write observations and supervision to physically separate NPZ files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset.validate()
    np.savez_compressed(directory / "observations.npz", x=dataset.x, observed=dataset.observed,
                        subject_id=dataset.subject_id, case_id=dataset.case_id,
                        scenario=dataset.scenario, condition=dataset.condition, fs=dataset.fs)
    labels = {"y": dataset.y, "corruption": dataset.corruption, "severity": dataset.severity}
    if dataset.latent_hr is not None:
        labels.update(latent_hr=dataset.latent_hr, latent_rr=dataset.latent_rr)
    np.savez_compressed(directory / "labels.npz", **labels)
    manifest = {"schema_version": 1, "n_cases": len(dataset), "n_subjects": len(np.unique(dataset.subject_id)),
                "modalities": list(MODALITIES), "target_names": ["hr_bpm", "rr_bpm"],
                "fs": dataset.fs, "duration_s": dataset.x.shape[2] / dataset.fs,
                "provenance": "Independent fully synthetic equations; no patient/UCLA data or source code",
                "sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                           for name in ("observations.npz", "labels.npz")}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_dataset(directory: str | Path, verify: bool = True) -> PhysioDataset:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if verify:
        for name, digest in manifest["sha256"].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Checksum mismatch: {name}")
    with np.load(directory/"observations.npz", allow_pickle=False) as obs, np.load(directory/"labels.npz", allow_pickle=False) as labels:
        values = {name: obs[name].copy() for name in obs.files}
        values.update({name: labels[name].copy() for name in labels.files})
    values["fs"] = float(values["fs"])
    dataset = PhysioDataset(**values)
    dataset.validate()
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=180)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    dataset = generate_dataset(n_subjects=args.subjects, windows_per_subject=args.windows, seed=args.seed)
    splits = split_subjects(dataset, seed=args.seed)
    report = {name: save_dataset(part, args.output/name) for name, part in splits.items()}
    print(json.dumps({name: {"cases": value["n_cases"], "subjects": value["n_subjects"]} for name, value in report.items()}, indent=2))


if __name__ == "__main__":
    main()
