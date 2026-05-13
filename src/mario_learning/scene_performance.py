"""Per-scene performance metrics and trajectory figures.

For each scene, aggregate Cleared / Duration / Hits_taken across repetitions
and emit a per_scene table. For each (Subject, Level), render a static facet
grid showing each scene's performance across repetitions over time — this is
the static replacement for the ipywidgets-driven NB_scene_performance plot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mario_learning import plots, provenance

log = logging.getLogger(__name__)

METRICS = ["Cleared", "Duration", "Hits_taken", "X_Traveled"]


def run(
    clips: pd.DataFrame,
    out_dir: Path,
    *,
    parameters: dict,
    inputs: list,
    cfg: dict,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    per_scene = _per_scene_table(clips)
    per_scene_path = out_dir / "per_scene.csv"
    per_scene.to_csv(per_scene_path, index=False)
    provenance.write_sidecar(per_scene_path, parameters=parameters, inputs=inputs)
    paths: dict[str, Path] = {"table": per_scene_path}

    for (subject, level), g in clips.groupby(["Subject", "Level"]):
        fig_path = figs_dir / f"sub-{subject}_{level}_scene_perf.png"
        _figure_subject_level(subject, level, g, fig_path, cfg=cfg)
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        paths[f"figure_sub-{subject}_{level}"] = fig_path

    log.info("scene_performance: %d (subject, level) combos", clips[["Subject", "Level"]].drop_duplicates().shape[0])
    return paths


def _per_scene_table(clips: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in METRICS if m in clips.columns]
    g = clips.groupby(["Subject", "Level", "SceneID"])
    table = g[metrics].mean(numeric_only=True).reset_index()
    table["n_clips"] = g.size().values
    return table


def _figure_subject_level(subject: str, level: str, g: pd.DataFrame, out_path: Path, *, cfg: dict) -> None:
    """One subplot per scene in the level; x = repetition index, y = Cleared (binary) with Duration as size."""
    style = plots.style(cfg)
    g = g.sort_values("ClipCode").copy()
    g["rep_idx"] = g.groupby("SceneID").cumcount()
    scenes = sorted(g["SceneID"].unique(), key=lambda s: int(s.split("s")[-1]))
    n = len(scenes)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.2 * rows), squeeze=False, sharey=True)
    for i, scene in enumerate(scenes):
        ax = axes[i // cols][i % cols]
        scene_df = g[g["SceneID"] == scene]
        cleared = scene_df["Cleared"].to_numpy()
        rep = scene_df["rep_idx"].to_numpy()
        durations = scene_df["Duration"].fillna(0).to_numpy()
        ax.scatter(rep, cleared, s=20 + 12 * durations, c=cleared, cmap="RdYlGn", vmin=0, vmax=1, edgecolor="k", lw=0.4)
        ax.set_title(scene, fontsize=8)
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["death", "clear"], fontsize=7)
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis("off")
    fig.suptitle(f"sub-{subject} — {level} per-scene trajectories (size ∝ duration)")
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])
