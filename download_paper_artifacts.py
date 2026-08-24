#!/usr/bin/env python3
"""Download, verify, and safely extract the frozen paper-result bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import BinaryIO


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "paper_artifacts.json"
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
        destination.write(chunk)


def _download(url: str, target: Path, *, size: int, checksum: str, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == size and sha256_file(target) == checksum:
            return
        if not overwrite:
            raise ValueError(f"existing file failed validation: {target}; use --overwrite")
        target.unlink()
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() and not overwrite else 0
    if overwrite:
        partial.unlink(missing_ok=True)
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.URLError as error:
        raise RuntimeError(f"download failed for {url}: {error}") from error
    append = offset > 0 and getattr(response, "status", None) == 206
    mode = "ab" if append else "wb"
    with response, partial.open(mode) as handle:
        _copy_stream(response, handle)
    if partial.stat().st_size != size or sha256_file(partial) != checksum:
        raise ValueError(f"checksum or size mismatch after downloading {url}")
    os.replace(partial, target)


def _validate_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive path: {member.name}")
    if member.isdev() or member.isfifo():
        raise ValueError(f"unsupported special archive entry: {member.name}")
    if member.issym() or member.islnk():
        link = PurePosixPath(member.linkname)
        if link.is_absolute():
            raise ValueError(f"unsafe absolute archive link: {member.name}")
        resolved = path.parent.joinpath(link)
        depth = 0
        for part in resolved.parts:
            depth = depth - 1 if part == ".." else depth + (part not in {"", "."})
            if depth < 0:
                raise ValueError(f"archive link escapes extraction root: {member.name}")


def _safe_extract(archive: Path, project_root: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise ValueError(f"destination already exists: {destination}; use --overwrite")
        shutil.rmtree(destination)
    with tempfile.TemporaryDirectory(prefix=".otrr_extract_", dir=project_root) as temporary:
        temporary_root = Path(temporary)
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            for member in members:
                _validate_member(member)
            try:
                handle.extractall(temporary_root, members=members, filter="data")
            except TypeError:  # Python 3.10 lacks the extraction filter argument.
                handle.extractall(temporary_root, members=members)
        staged = temporary_root / destination.relative_to(project_root)
        if not staged.is_dir():
            raise ValueError(f"archive did not contain {destination.relative_to(project_root)}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, destination)


def install_bundle(bundle: dict, *, project_root: Path, cache: Path, overwrite: bool) -> None:
    parts = bundle.get("parts", [])
    if not parts or any(not part.get("url") for part in parts):
        raise RuntimeError(
            f"artifact {bundle['name']} has no public download URL in this manifest"
        )
    local_parts = []
    for part in parts:
        target = cache / str(part["filename"])
        _download(
            str(part["url"]),
            target,
            size=int(part["size"]),
            checksum=str(part["sha256"]),
            overwrite=overwrite,
        )
        local_parts.append(target)
    archive = cache / str(bundle["archive_filename"])
    if not (len(local_parts) == 1 and local_parts[0] == archive):
        temporary_archive = archive.with_suffix(archive.suffix + ".assembling")
        with temporary_archive.open("wb") as output:
            for part in local_parts:
                with part.open("rb") as source:
                    _copy_stream(source, output)
        os.replace(temporary_archive, archive)
    if archive.stat().st_size != int(bundle["compressed_size"]):
        raise ValueError(f"reassembled size mismatch for {bundle['name']}")
    if sha256_file(archive) != str(bundle["sha256"]):
        raise ValueError(f"reassembled checksum mismatch for {bundle['name']}")
    destination = project_root / str(bundle["extraction_destination"])
    _safe_extract(archive, project_root, destination, overwrite=overwrite)
    for required in bundle.get("required_paths", []):
        if not (destination / str(required)).exists():
            raise ValueError(f"extracted bundle is missing {required}: {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--bundle", action="append", help="Download only a named bundle")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    project_root = args.project_root.resolve()
    cache = (args.cache_dir or project_root / ".paper_artifact_cache").resolve()
    selected = set(args.bundle or [])
    bundles = [item for item in manifest["bundles"] if not selected or item["name"] in selected]
    missing = selected - {item["name"] for item in bundles}
    if missing:
        raise ValueError(f"unknown bundle names: {sorted(missing)}")
    for bundle in bundles:
        print(f"installing {bundle['name']} -> {bundle['extraction_destination']}", flush=True)
        install_bundle(bundle, project_root=project_root, cache=cache, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
