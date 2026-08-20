import json
from pathlib import Path

from gait_aqa.visualization.transfer_report import build_transfer_results_figure


def test_build_transfer_results_figure(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "summary.png"

    report_path.write_text(
        json.dumps(
            {
                "nested_policy_aggregate_predictions": [
                    {"true_policy_score": 25.0, "pred_policy_score": 30.0},
                    {"true_policy_score": 75.0, "pred_policy_score": 70.0},
                ],
                "nested_policy_aggregate_metrics": {"mae": 5.0},
                "nested_policy_aggregate_mean_baseline_metrics": {"mae": 20.0},
            }
        ),
        encoding="utf-8",
    )

    result = build_transfer_results_figure(
        report_path,
        output_path,
    )

    assert result == output_path
    assert output_path.stat().st_size > 1_000
