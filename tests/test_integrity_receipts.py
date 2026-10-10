"""Evidence checks fail on byte changes and paths outside a receipt directory."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


specification = importlib.util.spec_from_file_location(
    "artifact_verifier", Path(__file__).resolve().parents[1] / "scripts" / "verify_artifacts.py")
verifier = importlib.util.module_from_spec(specification)
specification.loader.exec_module(verifier)


@pytest.fixture
def evidence_repository(tmp_path):
    source = tmp_path / "src" / "physiobench"
    source.mkdir(parents=True)
    for name in verifier.NEURAL_SOURCES:
        (source / name).write_text("# frozen fixture\n")
    result = tmp_path / "results" / "study"
    result.mkdir(parents=True)
    (result / "outcome.json").write_text('{"measured": 1}\n')
    receipt = {"status": "completed", "artifacts_sha256": {"outcome.json": verifier.sha256(result / "outcome.json")},
               "source_sha256": {name: verifier.sha256(source / name) for name in verifier.NEURAL_SOURCES}}
    (result / "manifest.json").write_text(json.dumps(receipt))
    return tmp_path, result


def test_complete_receipt_passes_and_changed_evidence_fails(evidence_repository):
    root, result = evidence_repository
    assert verifier.verify(root, root / "results")["status"] == "passed"
    (result / "outcome.json").write_text('{"measured": 2}\n')
    checked = verifier.verify(root, root / "results")
    assert checked["status"] == "failed"
    assert any("Artifact mismatch" in error for error in checked["errors"])


def test_critical_source_change_fails_without_a_reproduction_receipt(evidence_repository):
    root, result = evidence_repository
    (root / "src" / "physiobench" / "training.py").write_text("# changed model\n")
    checked = verifier.verify(root, root / "results")
    assert checked["status"] == "failed"
    assert any("Executed source mismatch" in error for error in checked["errors"])


def test_receipt_rejects_directory_traversal(evidence_repository):
    root, result = evidence_repository
    receipt = json.loads((result / "manifest.json").read_text())
    receipt["artifacts_sha256"] = {"../../outside.json": "0" * 64}
    (result / "manifest.json").write_text(json.dumps(receipt))
    checked = verifier.verify(root, root / "results")
    assert checked["status"] == "failed"
    assert any("escapes" in error for error in checked["errors"])
