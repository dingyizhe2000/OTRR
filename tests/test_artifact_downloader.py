from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from download_paper_artifacts import install_bundle


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_download_verify_and_extract_file_url(tmp_path: Path) -> None:
    source = tmp_path / "source.tar.gz"
    payload = b"complete\n"
    with tarfile.open(source, "w:gz") as archive:
        info = tarfile.TarInfo("results/example/COMPLETE.marker")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    bundle = {
        "name": "example",
        "archive_filename": "example.tar.gz",
        "compressed_size": source.stat().st_size,
        "sha256": _sha(source),
        "extraction_destination": "results/example",
        "required_paths": ["COMPLETE.marker"],
        "parts": [
            {
                "filename": "example.part-001",
                "url": source.as_uri(),
                "size": source.stat().st_size,
                "sha256": _sha(source),
            }
        ],
    }
    project = tmp_path / "project"
    project.mkdir()
    install_bundle(bundle, project_root=project, cache=tmp_path / "cache", overwrite=False)
    assert (project / "results/example/COMPLETE.marker").read_bytes() == payload


def test_missing_public_url_fails(tmp_path: Path) -> None:
    bundle = {"name": "missing", "parts": []}
    with pytest.raises(RuntimeError, match="no public download URL"):
        install_bundle(bundle, project_root=tmp_path, cache=tmp_path / "cache", overwrite=False)


def test_public_manifest_is_complete_and_parts_sum_to_archive_size() -> None:
    manifest = json.loads((ROOT / "paper_artifacts.json").read_text())
    assert manifest["artifact_version"] is None
    assert manifest["availability"] == "not_publicly_hosted"
    assert manifest["release_url"] is None
    assert len(manifest["bundles"]) == 3
    for bundle in manifest["bundles"]:
        assert bundle["parts"]
        assert sum(int(part["size"]) for part in bundle["parts"]) == int(bundle["compressed_size"])
        assert all(part["url"] is None for part in bundle["parts"])
        assert len(str(bundle["sha256"])) == 64


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "unsafe.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        payload = b"bad"
        info = tarfile.TarInfo("../escape")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    bundle = {
        "name": "unsafe",
        "archive_filename": "unsafe.tar.gz",
        "compressed_size": source.stat().st_size,
        "sha256": _sha(source),
        "extraction_destination": "results/unsafe",
        "required_paths": [],
        "parts": [{"filename": "unsafe.part", "url": source.as_uri(), "size": source.stat().st_size, "sha256": _sha(source)}],
    }
    with pytest.raises(ValueError, match="unsafe archive path"):
        install_bundle(bundle, project_root=tmp_path, cache=tmp_path / "cache", overwrite=False)
