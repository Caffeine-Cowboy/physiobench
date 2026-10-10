"""Verify published result/checkpoint hashes and the sources actually used.

Only Python's standard library is required. Snapshot hashes of incidental
modules are reported separately from critical executed pipeline sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


NEURAL_SOURCES = {"data.py", "simulator.py", "training.py", "posttraining.py", "experiments.py"}
BASELINE_SOURCES = {"data.py", "simulator.py", "baselines.py", "metrics.py", "baseline_experiment.py",
                    "experiments.py", "posttraining.py", "training.py"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def within(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Receipt path escapes its directory: {relative}")
    return path


def combined_source_hash(source_directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_directory.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify(repository: Path, result_directory: Path) -> dict:
    source = repository / "src" / "physiobench"
    verified, errors, incidental_drift, reproduced_sources = [], [], [], []
    manifests = sorted(result_directory.rglob("manifest.json"))
    for manifest_path in manifests:
        receipt = json.loads(manifest_path.read_text())
        if receipt.get("status") != "completed":
            errors.append(f"Incomplete receipt: {manifest_path}")
            continue
        artifacts = receipt.get("artifacts_sha256", {})
        if not artifacts:
            errors.append(f"No artifact hashes: {manifest_path}")
        for name, expected in artifacts.items():
            try:
                path = within(manifest_path.parent, name)
            except ValueError as error:
                errors.append(str(error))
                continue
            if not path.is_file() or sha256(path) != expected:
                errors.append(f"Artifact mismatch: {path}")
            else:
                verified.append(str(path.relative_to(repository)) if path.is_relative_to(repository) else str(path))
        critical = BASELINE_SOURCES if receipt.get("feature_count") == 70 else NEURAL_SOURCES
        recorded = receipt.get("source_sha256", {})
        for name in sorted(critical):
            path = source / name
            if name not in recorded or not path.is_file() or sha256(path) != recorded[name]:
                # A metadata-only runner revision was added after the original
                # neural run. Its exact historical bytes and an independently
                # executed current-source numerical reproduction are preserved;
                # original outcome/checkpoint files are never rewritten.
                proof_path = manifest_path.parent / "source_reproduction.json"
                proof = json.loads(proof_path.read_text()) if proof_path.is_file() else {}
                snapshot_name = proof.get("historical_source_snapshots", {}).get(name)
                snapshot = within(manifest_path.parent, snapshot_name) if snapshot_name else None
                reproduction_manifest = manifest_path.parent / "current_source_reproduction_manifest.json"
                valid_proof = (proof.get("status") == "completed"
                    and proof.get("all_prediction_arrays_bitwise_equal") is True
                    and proof.get("all_model_tensors_bitwise_equal") is True
                    and proof.get("all_metrics_and_bootstraps_equal") is True
                    and proof.get("n_prediction_files") == 432 and proof.get("n_checkpoint_state_dicts") == 24
                    and proof.get("original_source_sha256", {}).get(name) == recorded.get(name)
                    and proof.get("current_source_sha256", {}).get(name) == sha256(path)
                    and snapshot is not None and snapshot.is_file() and sha256(snapshot) == recorded.get(name)
                    and reproduction_manifest.is_file()
                    and sha256(reproduction_manifest) == proof.get("reproduction_manifest_sha256"))
                if valid_proof:
                    reproduced_sources.append({"source": name, "receipt": str(proof_path),
                        "historical_hash": recorded[name], "current_hash": sha256(path),
                        "verification": "exact historical bytes plus current-source bitwise numerical reproduction"})
                else:
                    errors.append(f"Executed source mismatch: {path}")
        for name, expected in recorded.items():
            if name not in critical and (not (source / name).is_file() or sha256(source / name) != expected):
                incidental_drift.append({"receipt": str(manifest_path), "source": name,
                                         "reason": "Incidental module snapshot; outside this executed pipeline"})
    protocols = sorted(result_directory.rglob("*.protocol.json"))
    for protocol_path in protocols:
        protocol = json.loads(protocol_path.read_text())
        report_path = protocol_path.with_name(protocol_path.name.replace(".protocol.json", ".json"))
        if not report_path.is_file():
            errors.append(f"Missing frozen external report: {report_path}")
            continue
        report = json.loads(report_path.read_text())
        if report.get("frozen_protocol_sha256") != sha256(protocol_path):
            errors.append(f"External protocol mismatch: {protocol_path}")
        if protocol.get("source_code_sha256") != combined_source_hash(source):
            errors.append(f"External frozen source mismatch: {source}")
        for checkpoint, expected in protocol.get("checkpoint_sha256", {}).items():
            path = repository / "results" / "neural-study" / "checkpoints" / (checkpoint + ".pt")
            if not path.is_file() or sha256(path) != expected:
                errors.append(f"External frozen checkpoint mismatch: {path}")
            else:
                verified.append(str(path.relative_to(repository)))
        for checkpoint, value in report.get("models", {}).items():
            if checkpoint == "signal_processing":
                continue
            if value.get("checkpoint_sha256") != protocol.get("checkpoint_sha256", {}).get(checkpoint):
                errors.append(f"External report references a different checkpoint: {checkpoint}")
    if not manifests and not protocols:
        errors.append(f"No evidence receipts found: {result_directory}")
    return {"status": "passed" if not errors else "failed", "verified_hashes": len(verified),
            "manifests": len(manifests), "external_protocols": len(protocols),
            "errors": errors, "incidental_source_snapshot_drift": incidental_drift,
            "historical_sources_with_numeric_reproduction": reproduced_sources,
            "scope": "tracked model/result files and executed-source hashes; public raw files are downloaded separately"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", type=Path)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    results = arguments.results.resolve() if arguments.results else repository / "results"
    result = verify(repository, results)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
