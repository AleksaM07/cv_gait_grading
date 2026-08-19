"""Evaluation metrics for score regression and ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, R2, Spearman, and pairwise ranking accuracy."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    available = np.isfinite(true) & np.isfinite(pred)
    true = true[available]
    pred = pred[available]
    if true.size == 0:
        raise ValueError("No finite prediction pairs are available")
    error = pred - true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1.0 - np.sum(error**2) / denom) if denom > 0 else 0.0
    spearman = _rank_corr(true, pred)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
        "pairwise_ranking_accuracy": pairwise_ranking_accuracy(true, pred),
    }


def regression_table(predictions: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Build one metrics row per score target."""
    subset = predictions[predictions["split"] == split]
    if subset.empty:
        subset = predictions
    train = predictions[predictions["split"] == "train"]
    rows = []
    for column in predictions.columns:
        if not column.startswith("true_") or not column.endswith("_score"):
            continue
        target = column.removeprefix("true_")
        pred_column = f"pred_{target}"
        if pred_column not in predictions:
            continue
        true = subset[column].to_numpy(dtype=float)
        pred = subset[pred_column].to_numpy(dtype=float)
        if not (np.isfinite(true) & np.isfinite(pred)).any():
            continue
        row = {"target": target, "split": split}
        row.update(regression_metrics(true, pred))
        train_values = train[column].to_numpy(dtype=float) if not train.empty else true
        train_values = train_values[np.isfinite(train_values)]
        if train_values.size:
            baseline_prediction = np.full_like(true, train_values.mean())
            baseline = regression_metrics(true, baseline_prediction)
            row["baseline_mae"] = baseline["mae"]
            row["baseline_rmse"] = baseline["rmse"]
            row["mae_skill_vs_mean"] = (
                1.0 - row["mae"] / baseline["mae"] if baseline["mae"] > 0.0 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def classification_table(
    predictions: pd.DataFrame, split: str = "test"
) -> pd.DataFrame:
    """Build binary metrics while exposing one-class evaluation splits."""
    subset = predictions[predictions["split"] == split]
    if subset.empty:
        subset = predictions
    rows: list[dict[str, float | str]] = []
    for column in predictions.columns:
        if not column.startswith("true_") or not column.endswith("_label"):
            continue
        target = column.removeprefix("true_")
        pred_column = f"pred_{target}"
        if pred_column not in predictions:
            continue
        true = subset[column].to_numpy(dtype=float)
        score = subset[pred_column].to_numpy(dtype=float)
        available = np.isfinite(true) & np.isfinite(score)
        if not available.any():
            continue
        true = true[available]
        score = score[available]
        row: dict[str, float | str] = {
            "target": target,
            "split": split,
            "samples": float(true.size),
            "positive_rate": float(true.mean()),
            "classes_present": float(np.unique(true).size),
        }
        row.update(classification_metrics(true, score))
        rows.append(row)
    return pd.DataFrame(rows)


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Return precision, recall, and F1 for binary labels."""
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_score, dtype=float) >= threshold
    tp = float(np.sum((true == 1) & pred))
    fp = float(np.sum((true == 0) & pred))
    fn = float(np.sum((true == 1) & ~pred))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def pairwise_ranking_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Measure how often pairwise score order is preserved."""
    total = 0
    correct = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            true_diff = y_true[i] - y_true[j]
            pred_diff = y_pred[i] - y_pred[j]
            if np.isclose(true_diff, 0.0):
                continue
            total += 1
            correct += int(np.sign(true_diff) == np.sign(pred_diff))
    return float(correct / total) if total else 0.0


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    if np.isclose(ar.std(), 0.0) or np.isclose(br.std(), 0.0):
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])
