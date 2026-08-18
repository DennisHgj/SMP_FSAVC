"""Deterministic N-way K-shot task sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_annotations(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] < 3:
        raise ValueError(f"Expected at least three CSV columns in {path}")
    frame = frame.iloc[:, :3].copy()
    frame.columns = ["clip_id", "label", "prompt"]
    return frame


def sample_classes(
    annotations: pd.DataFrame,
    n_way: int,
    rng: np.random.Generator,
) -> list[int]:
    classes = np.sort(annotations["label"].unique())
    if n_way > len(classes):
        raise ValueError(f"Requested {n_way} classes, but only {len(classes)} exist")
    return sorted(rng.choice(classes, size=n_way, replace=False).tolist())


def sample_few_shot_task(
    support_pool: pd.DataFrame,
    query_pool: pd.DataFrame,
    selected_classes: list[int],
    k_shot: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    """Sample support examples and remap labels to the task-local range."""
    if k_shot < 1:
        raise ValueError("k_shot must be positive")
    class_map = {int(label): index for index, label in enumerate(selected_classes)}
    support_parts = []
    for label in selected_classes:
        candidates = support_pool[support_pool["label"] == label]
        if len(candidates) < k_shot:
            raise ValueError(
                f"Class {label} contains {len(candidates)} support samples; "
                f"{k_shot} are required"
            )
        indices = rng.choice(candidates.index.to_numpy(), size=k_shot, replace=False)
        support_parts.append(candidates.loc[indices])

    support = pd.concat(support_parts, ignore_index=True)
    query = query_pool[query_pool["label"].isin(selected_classes)].copy()
    if query.empty:
        raise ValueError("The selected task has no query examples")
    support["label"] = support["label"].map(class_map)
    query["label"] = query["label"].map(class_map)
    return support, query.reset_index(drop=True), class_map
