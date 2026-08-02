"""Plot generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_score_over_time(scores: np.ndarray, path: str | Path) -> Path:
    """Save a score-over-time plot."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(len(scores)), scores, color="#1f77b4")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Score")
    ax.set_title("Estimated gait quality over time")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_heatmap(values: np.ndarray, path: str | Path) -> Path:
    """Save a temporal irregularity heatmap."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.imshow(values[None, :], aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xlabel("Frame")
    ax.set_title("Temporal irregularity severity")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
