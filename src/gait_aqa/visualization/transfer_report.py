"""Generate report figures from the frozen video-transfer experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_transfer_results_figure(
    report_path: Path,
    output_path: Path,
) -> Path:
    """Build a nested policy-level scatter plot and baseline comparison.

    Args:
        report_path: Transfer training report JSON.
        output_path: Destination PNG path.

    Returns:
        Path to the generated PNG.

    Raises:
        FileNotFoundError: If a required experiment artifact is missing.
        ValueError: If the report lacks finite policy predictions.
    """
    if not report_path.is_file():
        raise FileNotFoundError(f"Training report not found: {report_path}")

    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    policy_rows = report.get("nested_policy_aggregate_predictions", [])
    if len(policy_rows) < 2:
        raise ValueError("Training report needs at least two policy predictions")
    actual = np.asarray([row["true_policy_score"] for row in policy_rows], dtype=float)
    predicted = np.asarray(
        [row["pred_policy_score"] for row in policy_rows], dtype=float
    )
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Policy predictions contain non-finite values")

    transfer_mae = float(report["nested_policy_aggregate_metrics"]["mae"])
    mean_mae = float(report["nested_policy_aggregate_mean_baseline_metrics"]["mae"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    scatter_axis, bar_axis = axes

    scatter_axis.scatter(
        actual,
        predicted,
        color="#16697a",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    score_min = float(min(actual.min(), predicted.min())) - 3.0
    score_max = float(max(actual.max(), predicted.max())) + 3.0
    scatter_axis.plot(
        [score_min, score_max],
        [score_min, score_max],
        linestyle="--",
        color="#555555",
        linewidth=1.2,
        label="idealno",
    )
    scatter_axis.set_xlim(score_min, score_max)
    scatter_axis.set_ylim(score_min, score_max)
    scatter_axis.set_xlabel("Stvarni prose\u010dni policy skor")
    scatter_axis.set_ylabel("Held-out predikcija policy skora")
    scatter_axis.set_title("Nested CV: devet nevi\u0111enih politika")
    scatter_axis.grid(alpha=0.2)
    scatter_axis.legend(frameon=False)

    names = ["R3D-18\ntransfer", "Leave-one-policy\nmean"]
    values = [transfer_mae, mean_mae]
    colors = ["#16697a", "#d1495b"]
    bars = bar_axis.bar(names, values, color=colors, width=0.68)
    bar_axis.set_ylabel("Nested policy-level MAE")
    bar_axis.set_title("Policy prosek: manje je bolje")
    bar_axis.set_ylim(0.0, max(values) * 1.22)
    bar_axis.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        bar_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle("Ocena MuJoCo politika iz side-view RGB videa")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    """Generate the transfer-results report figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/predictions/r3d18_transfer_side_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/predictions/transfer_results_summary.png"),
    )
    args = parser.parse_args()
    output = build_transfer_results_figure(
        args.report,
        args.output,
    )
    print(output)


if __name__ == "__main__":
    main()
