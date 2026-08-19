"""Import exported metrics from the MuJoCo walker repository."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def import_walker_outputs(
    walker_repo: str | Path,
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Import existing walker CSV exports and create a CV-training label table."""
    repo = Path(walker_repo)
    analysis_dir = repo / "analysis_outputs"
    policy_path = analysis_dir / "policy_metrics.csv"
    selected_path = analysis_dir / "selected_checkpoints.csv"
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing walker export: {policy_path}")
    policies = pd.read_csv(policy_path)
    required = {
        "policy_id",
        "run_name",
        "checkpoint_step",
        "scenario",
        "composite_score",
        "score_stability",
        "score_tracking",
        "score_smoothness",
        "score_upright",
    }
    missing = required - set(policies.columns)
    if missing:
        raise ValueError(f"Walker policy metrics are missing: {sorted(missing)}")
    selected = pd.read_csv(selected_path) if selected_path.exists() else pd.DataFrame()
    if not selected.empty:
        keep = ["policy_id", "xml_path_saved", "reference_gait", "command_profile"]
        policies = policies.merge(
            selected[[column for column in keep if column in selected]],
            on="policy_id",
            how="left",
        )

    rows: list[dict[str, object]] = []
    for _, policy in policies.iterrows():
        clip_id = f"{policy['policy_id']}_{policy['scenario']}"
        falls = float(policy.get("falls", 0.0) or 0.0)
        rows.append(
            {
                "clip_id": clip_id,
                "video_path": "",
                "policy_run": policy["run_name"],
                "checkpoint_step": int(policy["checkpoint_step"]),
                "xml_model": policy.get("xml_path_saved", ""),
                "scenario": policy["scenario"],
                "seed": "",
                "camera": "unrendered",
                "fps": 0,
                "frame_count": 0,
                "duration_seconds": float("nan"),
                "split_group": (
                    f"{policy['policy_id']}:{int(policy['checkpoint_step'])}"
                ),
                "overall_score": 100.0 * float(policy["composite_score"]),
                "stability_score": 100.0 * float(policy["score_stability"]),
                "contact_score": float("nan"),
                "symmetry_score": float("nan"),
                "periodicity_score": float("nan"),
                "smoothness_score": 100.0 * float(policy["score_smoothness"]),
                "tracking_score": 100.0 * float(policy["score_tracking"]),
                "upright_score": 100.0 * float(policy["score_upright"]),
                "foot_sliding_label": float("nan"),
                "hopping_label": float("nan"),
                "micro_stepping_label": float("nan"),
                "asymmetry_label": float("nan"),
                "torso_instability_label": float("nan"),
                "toe_dragging_label": float("nan"),
                "fall_label": int(falls > 0.0),
                "command_ignoring_label": float("nan"),
                "telemetry_path": str(policy_path),
            }
        )
    imported = pd.DataFrame(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    imported.to_csv(output / "walker_metric_labels.csv", index=False)
    return imported
