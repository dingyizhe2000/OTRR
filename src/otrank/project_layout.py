"""Resolve repository paths in standalone and development checkouts."""

from __future__ import annotations

from pathlib import Path


def code_root(project_root: Path) -> Path:
    """Return the directory containing ``src/otrank``."""

    root = project_root.resolve()
    if (root / "src" / "otrank").is_dir():
        return root
    if (root / "code" / "src" / "otrank").is_dir():
        return root / "code"
    raise FileNotFoundError(f"cannot locate src/otrank below {root}")


def project_root(code_directory: Path) -> Path:
    """Return the result-bundle root for a source checkout."""

    source = code_directory.resolve()
    parent = source.parent
    if source.name == "code" and (parent / "results").is_dir():
        return parent
    return source
