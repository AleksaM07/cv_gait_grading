"""Model persistence."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_model(model: Any, path: str | Path) -> Path:
    """Save a model artifact with pickle."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(model, file)
    return output


def load_model(path: str | Path) -> Any:
    """Load a model artifact."""
    model_path = Path(path)
    with model_path.open("rb") as file:
        return pickle.load(file)
