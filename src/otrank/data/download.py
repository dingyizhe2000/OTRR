"""Download helpers with tqdm progress bars."""

from __future__ import annotations

from pathlib import Path

import requests
from tqdm import tqdm


def download_file(
    url: str,
    output_path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
    timeout: float = 30.0,
    overwrite: bool = False,
    desc: str | None = None,
) -> Path:
    """Stream a URL to disk with a byte-level progress bar.

    This is intended for large public datasets distributed as archives. It does
    not silently overwrite existing files unless ``overwrite=True``.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return output_path

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        label = desc or output_path.name
        with output_path.open("wb") as fh, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=label,
            dynamic_ncols=True,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                bar.update(len(chunk))
    return output_path
