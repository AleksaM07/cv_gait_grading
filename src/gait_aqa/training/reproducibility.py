"""Reproducibility helpers."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Hash a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: str | Path = ".") -> str:
    """Return the current Git commit or `unknown`."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(path),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()
