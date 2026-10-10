"""Compare a reproduced study with the original without changing its evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def compare(original: Path, reproduced: Path) -> dict:
    first_manifest = json.loads((original / "manifest.json").read_text())
    second_manifest = json.loads((reproduced / "manifest.json").read_text())
    if first_manifest.get("status") != "completed" or second_manifest.get("status") != "completed":
        raise ValueError("Both studies must have completed receipts")
    if first_manifest["configuration"] != second_manifest["configuration"]:
        raise ValueError("Different study configurations")
    if first_manifest["data_sha256"] != second_manifest["data_sha256"]:
        raise ValueError("Generated data differ")
    original_predictions = sorted((original / "predictions").glob("*.npz"))
    checkpoints = sorted((original / "checkpoints").glob("*.pt"))
    if {p.name for p in original_predictions} != {p.name for p in (reproduced / "predictions").glob("*.npz")}:
        raise ValueError("Prediction file identities differ")
    if {p.name for p in checkpoints} != {p.name for p in (reproduced / "checkpoints").glob("*.pt")}:
        raise ValueError("Checkpoint identities differ")
    for path in original_predictions:
        with np.load(path, allow_pickle=False) as first, np.load(reproduced / "predictions" / path.name, allow_pickle=False) as second:
            if set(first.files) != set(second.files):
                raise ValueError(f"Prediction fields differ: {path.name}")
            for key in first.files:
                np.testing.assert_array_equal(first[key], second[key], err_msg=f"{path.name}:{key}")
    for path in checkpoints:
        first = torch.load(path, map_location="cpu", weights_only=True)
        second = torch.load(reproduced / "checkpoints" / path.name, map_location="cpu", weights_only=True)
        if first.keys() != second.keys() or first["state_dict"].keys() != second["state_dict"].keys():
            raise ValueError(f"Checkpoint fields differ: {path.name}")
        for key in first["state_dict"]:
            if not torch.equal(first["state_dict"][key], second["state_dict"][key]):
                raise ValueError(f"Model tensors differ: {path.name}:{key}")
        for key in first:
            if key != "state_dict" and first[key] != second[key]:
                raise ValueError(f"Checkpoint metadata differ: {path.name}:{key}")
    for name in ("per_run_metrics.json", "aggregate_metrics.json", "paired_subject_bootstrap.json"):
        if json.loads((original / name).read_text()) != json.loads((reproduced / name).read_text()):
            raise ValueError(f"Metrics/bootstraps differ: {name}")
    return {"status": "completed", "comparison": "bitwise numerical reproduction",
            "n_prediction_files": len(original_predictions), "n_checkpoint_state_dicts": len(checkpoints),
            "all_prediction_arrays_bitwise_equal": True, "all_model_tensors_bitwise_equal": True,
            "all_metrics_and_bootstraps_equal": True,
            "original_source_sha256": first_manifest["source_sha256"],
            "reproduced_source_sha256": second_manifest["source_sha256"],
            "data_sha256": first_manifest["data_sha256"],
            "scope": "same-platform bitwise equality; failures on other platforms may require a declared numerical tolerance"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("Choose a fresh output path; do not overwrite comparison evidence")
    result = compare(arguments.original, arguments.reproduction)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "n_prediction_files", "n_checkpoint_state_dicts")}))


if __name__ == "__main__":
    main()
