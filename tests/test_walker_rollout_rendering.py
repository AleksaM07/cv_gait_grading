import json
from pathlib import Path

import numpy as np
import pandas as pd

from gait_aqa.reference_videos.render_walker_rollouts import (
    CAMERAS,
    SCENARIOS,
    _scaled_camera_distance,
    _scaled_lookat_floor,
    apply_reference_scores,
    build_rollout_tasks,
    discover_policies,
)


def test_discover_policies_selects_best_available_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "biomechanics_example"
    checkpoint_root = run_dir / "checkpoints"
    for step in (100, 200, 300):
        (checkpoint_root / f"{step:012d}").mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "env": {
                    "command_profile": "standard",
                    "reference_gait": "none",
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "train.log").write_text(
        "eval | step=100 | reward=2.0\n"
        "eval | step=200 | reward=7.0\n"
        "eval | step=300 | reward=5.0\n",
        encoding="utf-8",
    )

    policies = discover_policies(tmp_path)

    assert len(policies) == 1
    assert policies[0]["checkpoint_step"] == 200
    assert policies[0]["selection_reason"] == "max_logged_eval_reward"


def test_default_plan_has_unique_rollouts_and_two_views(tmp_path: Path) -> None:
    policy = {
        "policy_id": "P01_test",
        "run_name": "test_run",
        "policy_type": "biomechanics",
        "checkpoint_step": 10,
        "checkpoint_path": "checkpoint",
        "training_reward": 1.0,
        "selection_reason": "test",
        "command_profile": "standard",
        "reference_gait": "none",
        "run_config_path": "config.json",
    }

    tasks = build_rollout_tasks([policy], tmp_path)

    assert len(tasks) == len(SCENARIOS)
    assert len({task["rollout_id"] for task in tasks}) == len(tasks)
    clip_ids = {
        f"{task['rollout_id']}__{camera.name}" for task in tasks for camera in CAMERAS
    }
    assert len(clip_ids) == len(SCENARIOS) * len(CAMERAS)


def test_camera_distance_tracks_model_extent() -> None:
    assert np.isclose(_scaled_camera_distance(4.2, 2.2), 4.2)
    assert np.isclose(_scaled_camera_distance(4.2, 0.8), 4.2 * 0.8 / 2.2)
    assert _scaled_camera_distance(4.2, 0.01) == 0.8
    assert _scaled_camera_distance(4.2, float("nan")) == 4.2
    assert np.isclose(_scaled_lookat_floor(2.2), 0.70)
    assert np.isclose(_scaled_lookat_floor(0.8), 0.70 * 0.8 / 2.2)


def test_scores_are_computed_once_per_rollout_not_per_view() -> None:
    rows = []
    for rollout_id, stability, tracking, upright, smoothness in (
        ("best", 1.0, 0.1, 0.95, 1.0),
        ("worst", 0.5, 0.7, 0.50, 3.0),
    ):
        for camera in ("side", "front_oblique"):
            rows.append(
                {
                    "rollout_id": rollout_id,
                    "policy_id": rollout_id,
                    "checkpoint_step": 1,
                    "scenario": "forward",
                    "seed": 7,
                    "camera": camera,
                    "status": "recorded",
                    "mean_first_fall_survival_fraction": stability,
                    "tracking_rmse": tracking,
                    "mean_torso_up": upright,
                    "mean_action_rate_norm": smoothness,
                }
            )

    scored = apply_reference_scores(pd.DataFrame(rows))

    best = scored[scored["rollout_id"] == "best"]
    worst = scored[scored["rollout_id"] == "worst"]
    assert np.allclose(best["composite_score"], 1.0)
    assert np.allclose(worst["composite_score"], 0.0)
    assert best["composite_score"].nunique() == 1
