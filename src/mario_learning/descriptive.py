"""Descriptive statistics for a clips DataFrame.

Emits four tables + a small QC panel:

- ``subjects.csv``  — per-subject session/run/clip counts and overall completion
- ``scenes.csv``    — per-scene clip counts, subject coverage, completion rate
- ``phases.csv``    — per-(subject, phase) clip counts and completion rates
- ``levels.csv``    — per-level clip counts and completion rates
- ``desc-qc.png``   — three-panel visual summary
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mario_learning import plots, provenance

log = logging.getLogger(__name__)


def run(
    clips: pd.DataFrame,
    out_dir: Path,
    *,
    parameters: dict,
    inputs: list,
    cfg: dict,
) -> dict[str, Path]:
    """Compute descriptive tables + QC figure. Returns the produced paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "subjects": _per_subject(clips),
        "scenes": _per_scene(clips),
        "phases": _per_phase(clips),
        "levels": _per_level(clips),
    }
    paths: dict[str, Path] = {}
    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        provenance.write_sidecar(path, parameters=parameters, inputs=inputs)
        paths[name] = path
        log.info("  %s → %d rows", path.name, len(df))

    qc_path = out_dir / "desc-qc.png"
    _qc_figure(clips, tables, qc_path, cfg=cfg)
    provenance.write_sidecar(qc_path, parameters=parameters, inputs=inputs)
    paths["qc"] = qc_path
    return paths


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _per_subject(clips: pd.DataFrame) -> pd.DataFrame:
    g = clips.groupby("Subject")
    n_runs = (
        clips[["Subject", "Session", "Run"]]
        .drop_duplicates()
        .groupby("Subject")
        .size()
    )
    rows = pd.DataFrame({
        "n_sessions": g["Session"].nunique(),
        "n_runs": n_runs,
        "n_clips": g.size(),
        "n_scenes_visited": g["SceneID"].nunique(),
        "completion_rate": g["Cleared"].mean(),
        "median_duration_s": g["Duration"].median(),
    }).reset_index()
    return rows


def _per_scene(clips: pd.DataFrame) -> pd.DataFrame:
    g = clips.groupby("SceneID")
    rows = pd.DataFrame({
        "n_clips": g.size(),
        "n_subjects": g["Subject"].nunique(),
        "completion_rate": g["Cleared"].mean(),
        "median_duration_s": g["Duration"].median(),
        "median_hits": g["Hits_taken"].median(),
    }).reset_index()
    return rows.sort_values("SceneID")


def _per_phase(clips: pd.DataFrame) -> pd.DataFrame:
    g = clips.groupby(["Subject", "Phase"])
    rows = pd.DataFrame({
        "n_clips": g.size(),
        "completion_rate": g["Cleared"].mean(),
        "median_duration_s": g["Duration"].median(),
    }).reset_index()
    return rows


def _per_level(clips: pd.DataFrame) -> pd.DataFrame:
    g = clips.groupby("Level")
    rows = pd.DataFrame({
        "n_clips": g.size(),
        "n_scenes": g["SceneID"].nunique(),
        "n_subjects": g["Subject"].nunique(),
        "completion_rate": g["Cleared"].mean(),
    }).reset_index().sort_values("Level")
    return rows


# ---------------------------------------------------------------------------
# QC figure
# ---------------------------------------------------------------------------

def _qc_figure(
    clips: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    out_path: Path,
    *,
    cfg: dict,
) -> None:
    style = plots.style(cfg)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    subs = tables["subjects"].sort_values("Subject")
    ax.bar(subs["Subject"], subs["n_clips"], color="steelblue")
    ax.set_title("Clips per subject")
    ax.set_xlabel("Subject")
    ax.set_ylabel("# clips")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1]
    ax.hist(tables["scenes"]["completion_rate"], bins=20, color="darkorange")
    ax.set_title("Per-scene completion rate")
    ax.set_xlabel("Completion rate")
    ax.set_ylabel("# scenes")
    ax.set_xlim(0, 1)

    ax = axes[2]
    phase = tables["phases"]
    pivot = phase.pivot(index="Subject", columns="Phase", values="completion_rate").fillna(0)
    pivot.plot(kind="bar", ax=ax, colormap="viridis", legend=True)
    ax.set_title("Completion by phase")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Completion rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)

    n = len(clips)
    fig.suptitle(f"{clips['dataset'].iloc[0]} — {n} clips, {clips['Subject'].nunique()} subjects")
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])
