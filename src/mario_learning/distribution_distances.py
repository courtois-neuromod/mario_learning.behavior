"""Wasserstein-1 (earth-mover's) distances between performance distributions.

Two modes:

- **scalar** — for each configured variable (Duration, X_Traveled, Hits_taken),
  compute the 1-D Wasserstein distance between distributions of that variable
  across two groups within the dataset. Groups are defined by the
  ``grouping`` config option:
  - ``stage``      → 4-stage split per subject (early/late × discovery/practice)
    — pairwise EMD across all (Subject, Stage) combinations. **Recommended**
    (see [[feedback-four-stage-split]]).
  - ``subject_stage`` → like ``stage`` but only stages within a subject are
    paired — keeps cross-subject pairs out (per [[feedback-no-subject-averaging]]).
  - ``phase``      → discovery vs practice (subjects pooled)
  - ``subject``    → all pairwise subject distances
  - ``early_late`` → first half vs second half by ClipCode (per subject)
  - ``dataset``    → trivial single-group (always 0); useful for the compare
    task which pairs across datasets.

- **scene_space** — for each group, build a vector indexed by scene_id whose
  entries are the group's per-scene **failure rate** (``1 − Cleared``). The
  Wasserstein distance between two such vectors uses a scene-space ground
  metric: Euclidean in UMAP coords (from the ``clustering`` task's cache) or
  Jaccard on scene patterns (fallback). Implemented via the ``POT`` package
  (network simplex linear program over the cost matrix).

Outputs:
- ``scalar.csv``       — (variable, group_a, group_b, n_a, n_b, wasserstein)
- ``scene_space.csv``  — (group_a, group_b, ground_metric, wasserstein)
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import ot
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance

from mario_learning import provenance, utils

log = logging.getLogger(__name__)


def run(
    clips: pd.DataFrame,
    out_dir: Path,
    *,
    parameters: dict,
    inputs: list,
    cfg: dict,
    umap_coords_path: Path | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dd_cfg = cfg["analysis"]["distribution_distances"]
    grouping = dd_cfg["grouping"]
    variables = list(dd_cfg["scalar_variables"])
    ground_metric_cfg = dd_cfg["scene_space"].get("ground_metric", "umap")

    groups = _make_groups(clips, grouping)
    if len(groups) < 2:
        log.warning("distribution_distances: only %d group(s) for grouping=%s; skipping pairwise computation",
                    len(groups), grouping)
    within_subject = grouping == "subject_stage"

    scalar_rows = _scalar_pairs(clips, groups, variables, within_subject=within_subject)
    scene_rows, ground_metric_used = _scene_space_pairs(
        clips, groups, ground_metric_cfg, umap_coords_path, within_subject=within_subject,
    )

    scalar_path = out_dir / "scalar.csv"
    pd.DataFrame(scalar_rows).to_csv(scalar_path, index=False)
    provenance.write_sidecar(scalar_path, parameters=parameters, inputs=inputs)

    scene_path = out_dir / "scene_space.csv"
    pd.DataFrame(scene_rows).to_csv(scene_path, index=False)
    provenance.write_sidecar(scene_path, parameters=parameters, inputs=inputs)

    log.info("distribution_distances: %d scalar pairs, %d scene-space pairs (ground=%s)",
             len(scalar_rows), len(scene_rows), ground_metric_used)
    return {"scalar": scalar_path, "scene_space": scene_path}


def _make_groups(clips: pd.DataFrame, grouping: str) -> dict[str, pd.DataFrame]:
    if grouping == "phase":
        return {phase: g for phase, g in clips.groupby("Phase")}
    if grouping == "subject":
        return {f"sub-{s}": g for s, g in clips.groupby("Subject")}
    if grouping == "early_late":
        groups: dict[str, pd.DataFrame] = {}
        for subject, sub_df in clips.groupby("Subject"):
            sub_df = sub_df.sort_values("ClipCode")
            mid = len(sub_df) // 2
            groups[f"sub-{subject}_early"] = sub_df.iloc[:mid]
            groups[f"sub-{subject}_late"] = sub_df.iloc[mid:]
        return groups
    if grouping in ("stage", "subject_stage"):
        staged = utils.add_stage_column(clips)
        groups = {}
        for (subject, stage), g in staged.groupby(["Subject", "Stage"], observed=True):
            groups[f"sub-{subject}_{stage}"] = g
        return groups
    if grouping == "dataset":
        return {clips["dataset"].iloc[0]: clips}
    raise ValueError(f"Unknown grouping: {grouping}")


def _scalar_pairs(
    clips: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
    variables: list[str],
    *,
    within_subject: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for var in variables:
        if var not in clips.columns:
            log.warning("scalar: variable %s not in clips DataFrame", var)
            continue
        for a, b in combinations(groups, 2):
            if within_subject and not _same_subject(a, b):
                continue
            va = pd.to_numeric(groups[a][var], errors="coerce").dropna()
            vb = pd.to_numeric(groups[b][var], errors="coerce").dropna()
            if va.empty or vb.empty:
                continue
            rows.append({
                "variable": var,
                "group_a": a,
                "group_b": b,
                "n_a": len(va),
                "n_b": len(vb),
                "wasserstein": float(wasserstein_distance(va, vb)),
            })
    return rows


def _same_subject(a: str, b: str) -> bool:
    return _subject_key(a) == _subject_key(b)


def _subject_key(label: str) -> str | None:
    if label.startswith("sub-"):
        return label.split("_", 1)[0]
    return None


def _scene_space_pairs(
    clips: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
    ground_metric_cfg: str,
    umap_coords_path: Path | None,
    *,
    within_subject: bool = False,
) -> tuple[list[dict], str]:
    cost, scene_index, ground_metric_used = _scene_cost_matrix(ground_metric_cfg, umap_coords_path)
    rows: list[dict] = []
    failure_vectors = {name: _failure_vector(g, scene_index) for name, g in groups.items()}
    for a, b in combinations(groups, 2):
        if within_subject and not _same_subject(a, b):
            continue
        fa = failure_vectors[a]
        fb = failure_vectors[b]
        if fa.sum() == 0 or fb.sum() == 0:
            continue
        a_norm = fa / fa.sum()
        b_norm = fb / fb.sum()
        w = float(ot.emd2(a_norm, b_norm, cost))
        rows.append({
            "group_a": a,
            "group_b": b,
            "ground_metric": ground_metric_used,
            "wasserstein": w,
            "support_a": int((fa > 0).sum()),
            "support_b": int((fb > 0).sum()),
        })
    return rows, ground_metric_used


def _scene_cost_matrix(
    ground_metric_cfg: str,
    umap_coords_path: Path | None,
) -> tuple[np.ndarray, list[str], str]:
    if ground_metric_cfg == "umap" and umap_coords_path and umap_coords_path.exists():
        coords = pd.read_csv(umap_coords_path, index_col=0)
        scenes = coords.index.tolist()
        cost = squareform(pdist(coords.values, metric="euclidean"))
        return cost, scenes, "umap"
    # Fallback (or explicit jaccard request): Jaccard on Zenodo annotations.
    patterns = utils.scene_pattern_matrix()
    X = patterns.values.astype(bool)
    cost = squareform(pdist(X, metric="jaccard"))
    is_empty = ~X.any(axis=1)
    if is_empty.any():
        cost[is_empty, :] = 1.0
        cost[:, is_empty] = 1.0
        ee = np.outer(is_empty, is_empty)
        cost[ee] = 0.0
    np.fill_diagonal(cost, 0.0)
    return cost, list(patterns.index), "jaccard"


def _failure_vector(g: pd.DataFrame, scene_index: list[str]) -> np.ndarray:
    """Per-scene failure rate (1 − mean Cleared), aligned to scene_index. Zero for unobserved scenes."""
    rates = g.groupby("SceneID")["Cleared"].mean()
    failure = (1.0 - rates).reindex(scene_index).fillna(0.0)
    n = g.groupby("SceneID").size().reindex(scene_index).fillna(0)
    # Weight by observation count so under-sampled scenes don't dominate.
    return failure.to_numpy() * np.sqrt(n.to_numpy())
