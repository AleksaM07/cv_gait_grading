"""Minimal static dashboard writer."""

from __future__ import annotations

from pathlib import Path


def write_dashboard(path: str | Path, title: str, body: str) -> Path:
    """Write a simple HTML dashboard."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"<!doctype html><meta charset='utf-8'><title>{title}</title><h1>{title}</h1>{body}",
        encoding="utf-8",
    )
    return output
