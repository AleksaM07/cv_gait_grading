"""Import exported metrics from the MuJoCo walker repository."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gait_aqa.labels.score_components import score_from_walker_trial


def import_walker_outputs(
    walker_repo: str | Path,
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Import existing walker CSV exports and create a CV-training label table."""
    repo = Path(walker_repo)
    analysis_dir = repo / "analysis_outputs"
    trial_path = analysis_dir / "trial_metrics.csv"
    selected_path = analysis_dir / "selected_checkpoints.csv"
    if not trial_path.exists():
        raise FileNotFoundError(f"Missing walker export: {trial_path}")
    trials = pd.read_csv(trial_path)
    selected = pd.read_csv(selected_path) if selected_path.exists() else pd.DataFrame()
    if not selected.empty:
        keep = ["policy_id", "xml_path_saved", "reference_gait", "command_profile"]
        trials = trials.merge(selected[[c for c in keep if c in selected]], on="policy_id", how="left")

    rows: list[dict[str, object]] = []
    for _, trial in trials.iterrows():
        scores = score_from_walker_trial(trial)
        clip_id = f"{trial['policy_id']}_{trial['scenario']}_seed{int(trial['seed'])}"
        rows.append(
            {
                "clip_id": clip_id,
                "video_path": "",
                "policy_run": trial["run_name"],
                "checkpoint_step": int(trial["checkpoint_step"]),
                "xml_model": trial.get("xml_path_saved", ""),
                "scenario": trial["scenario"],
                "seed": int(trial["seed"]),
                "camera": "unrendered",
                "fps": 0,
                "frame_count": 0,
                "duration_seconds": trial.get("steps", 0),
                "split_group": f"{trial['policy_id']}:{int(trial['checkpoint_step'])}",
                "overall_score": scores["overall_score"],
                "stability_score": scores["stability"],
                "contact_score": scores["foot_contact_quality"],
                "symmetry_score": scores["left_right_symmetry"],
                "periodicity_score": scores["periodicity"],
                "smoothness_score": scores["smoothness"],
                "tracking_score": scores["command_tracking"],
                "foot_sliding_label": int(scores["irregularities"]["foot_sliding"] >= 0.35),
                "hopping_label": 0,
                "micro_stepping_label": 0,
                "asymmetry_label": 0,
                "torso_instability_label": int(scores["irregularities"]["torso_instability"] >= 0.40),
                "toe_dragging_label": 0,
                "fall_label": int(scores["irregularities"]["fall_or_near_fall"] >= 0.50),
                "command_ignoring_label": int(scores["irregularities"]["command_ignoring"] >= 0.45),
                "telemetry_path": str(trial_path),
            }
        )
    imported = pd.DataFrame(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    imported.to_csv(output / "walker_metric_labels.csv", index=False)
    return imported
