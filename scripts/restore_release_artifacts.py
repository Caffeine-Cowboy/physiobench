"""Restore only the 528 approved synthetic artifacts from the pinned release.

Uses Python's standard library. All entries and existing destinations are
validated before any extraction; sources, documentation and receipts are never
written from the bundle. Repeated execution verifies and preserves matching files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile


BUNDLE_SHA256 = "5bf4e66f3cfb40e9d86bdb3a678b64bf535f35a1ecd26bf0ddbdd916a4e67da5"
EXPECTED_ARTIFACT_COUNT = 528
ARCHIVE_PREFIX = "physiobench/"
BUNDLE_NAME = "release/physiobench-v0.1.0.zip"
MANIFEST_NAME = "release/artifacts-manifest.json"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_relative(name: str) -> PurePosixPath:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or "\x00" in name or name.startswith("/")):
        raise ValueError(f"Unsafe archive/manifest path: {name!r}")
    components = name.split("/")
    if any(part in ("", ".", "..") for part in components):
        raise ValueError(f"Unsafe archive/manifest path: {name!r}")
    return PurePosixPath(name)


def validate_member(info: zipfile.ZipInfo) -> None:
    # Directory records may end in '/', but cannot otherwise contain empty parts.
    safe_relative(info.filename[:-1] if info.is_dir() else info.filename)
    mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(mode):
        raise ValueError(f"Symlink archive member rejected: {info.filename}")
    kind = stat.S_IFMT(mode)
    if kind and kind not in (stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"Special archive member rejected: {info.filename}")
    if info.flag_bits & 1:
        raise ValueError("Encrypted archive members are not supported")


def reject_symlink_chain(repository: Path, destination: Path) -> None:
    for candidate in (destination, *destination.parents):
        if candidate == repository.parent:
            break
        if candidate.is_symlink():
            raise ValueError(f"Symlink destination/parent rejected: {candidate}")


def safe_destination(repository: Path, relative: str) -> Path:
    parts = safe_relative(relative).parts
    if parts[0] != "results" or PurePosixPath(relative).suffix not in (".pt", ".npz"):
        raise ValueError(f"Only approved results artifacts may be restored: {relative}")
    destination = repository.joinpath(*parts)
    reject_symlink_chain(repository, destination)
    if not destination.resolve().is_relative_to(repository.resolve()):
        raise ValueError(f"Destination escapes repository: {relative}")
    return destination


def expected_artifacts(repository: Path) -> dict[str, str]:
    path = repository / MANIFEST_NAME
    reject_symlink_chain(repository, path)
    value = json.loads(path.read_text())
    if (value.get("schema_version") != 1 or value.get("bundle") != BUNDLE_NAME
            or value.get("bundle_sha256") != BUNDLE_SHA256
            or value.get("archive_prefix") != ARCHIVE_PREFIX
            or value.get("n_synthetic_artifacts") != EXPECTED_ARTIFACT_COUNT):
        raise ValueError("Artifact manifest identity differs from the pinned release")
    expected = value.get("artifacts_sha256")
    if not isinstance(expected, dict) or len(expected) != EXPECTED_ARTIFACT_COUNT:
        raise ValueError("Artifact manifest has an incorrect expected path count")
    for name, checksum in expected.items():
        safe_destination(repository, name)
        if (not isinstance(checksum, str) or len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)):
            raise ValueError(f"Invalid expected checksum: {name}")
    return expected


def restore(repository: Path) -> dict:
    repository = repository.resolve()
    expected = expected_artifacts(repository)
    bundle = repository / BUNDLE_NAME
    reject_symlink_chain(repository, bundle)
    if bundle.is_symlink() or not bundle.is_file() or digest(bundle) != BUNDLE_SHA256:
        raise ValueError("Release bundle SHA-256 differs from the pinned original")
    payloads = {}
    with zipfile.ZipFile(bundle) as archive:
        members = archive.infolist()
        names = [info.filename for info in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member names are rejected")
        for info in members:
            validate_member(info)
        actual = {info.filename[len(ARCHIVE_PREFIX):] for info in members
                  if info.filename.startswith(ARCHIVE_PREFIX + "results/")
                  and PurePosixPath(info.filename).suffix in (".pt", ".npz")}
        if actual != set(expected):
            raise ValueError("Release contains a different synthetic artifact path set")
        # Authenticate every byte and reject conflicts before making any writes.
        for name, checksum in sorted(expected.items()):
            destination = safe_destination(repository, name)
            if destination.exists() and (not destination.is_file() or digest(destination) != checksum):
                raise FileExistsError(f"Refusing to replace a differing existing artifact: {name}")
            payload = archive.read(ARCHIVE_PREFIX + name)
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError(f"Artifact checksum mismatch inside pinned release: {name}")
            payloads[name] = payload
    written, unchanged = 0, 0
    for name, payload in payloads.items():
        destination = safe_destination(repository, name)
        if destination.exists():
            if not destination.is_file() or digest(destination) != expected[name]:
                raise FileExistsError(f"Artifact changed during restoration: {name}")
            unchanged += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        safe_destination(repository, name)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".restore-", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            # Hard-link creation is atomic and cannot replace a concurrently
            # created file. The temporary and destination share a filesystem.
            os.link(temporary, destination)
            written += 1
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"status": "passed", "bundle_sha256": BUNDLE_SHA256,
            "expected_artifacts": EXPECTED_ARTIFACT_COUNT,
            "restored_files": written, "already_matching_files": unchanged,
            "scope": "approved synthetic checkpoints/predictions only; no source or receipt overwrites"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(restore(args.repository), indent=2))


if __name__ == "__main__":
    main()
