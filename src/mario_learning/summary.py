"""Per-level grids of performance variables across scenes.

For each level, emit a figure of one panel per variable (Cleared, Duration,
Hits_taken) showing per-scene means with subject overlays, faceted by phase.
Mirrors the dense per-level figures from NB_summary.ipynb.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mario_learning import plots, provenance

log = logging.getLogger(__name__)

VARIABLES = ["Cleared", "Duration", "Hits_taken"]


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

    per_level = _aggregate(clips)
    csv_path = out_dir / "per_level_per_scene.csv"
    per_level.to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, parameters=parameters, inputs=inputs)

    paths: dict[str, Path] = {"table": csv_path}
    for level, level_df in per_level.groupby("Level"):
        fig_path = figs_dir / f"{level}_summary.png"
        _figure_for_level(level, level_df, fig_path, cfg=cfg)
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        paths[f"figure_{level}"] = fig_path
    log.info("summary: %d levels", per_level["Level"].nunique())
    return paths


def _aggregate(clips: pd.DataFrame) -> pd.DataFrame:
    """Per (Level, SceneID, Subject, Phase) mean of the summary variables."""
    cols = [c for c in VARIABLES if c in clips.columns]
    grouped = (
        clips.groupby(["Level", "SceneID", "Subject", "Phase"])[cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped["scene_order"] = grouped["SceneID"].str.extract(r"s(\d+)$").astype(int)
    return grouped.sort_values(["Level", "scene_order", "Subject", "Phase"])


def _figure_for_level(level: str, level_df: pd.DataFrame, out_path: Path, *, cfg: dict) -> None:
    style = plots.style(cfg)
    fig, axes = plt.subplots(len(VARIABLES), 1, figsize=(10, 2.6 * len(VARIABLES)), sharex=True)
    if len(VARIABLES) == 1:
        axes = [axes]
    scenes = level_df.sort_values("scene_order")["SceneID"].drop_duplicates().tolist()
    palette = plt.get_cmap("tab10")
    subjects = sorted(level_df["Subject"].unique())
    for i, var in enumerate(VARIABLES):
        ax = axes[i]
        for j, subject in enumerate(subjects):
            for phase, marker in [("discovery", "o"), ("practice", "x")]:
                sub_df = level_df[(level_df["Subject"] == subject) & (level_df["Phase"] == phase)]
                if sub_df.empty:
                    continue
                ax.plot(
                    sub_df["SceneID"],
                    sub_df[var],
                    marker=marker,
                    linestyle="-",
                    color=palette(j % 10),
                    label=f"sub-{subject} {phase}",
                    alpha=0.75,
                )
        ax.set_ylabel(var)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="best", ncol=2)
    axes[-1].set_xticks(range(len(scenes)))
    axes[-1].set_xticklabels(scenes, rotation=45, ha="right", fontsize=8)
    fig.suptitle(f"{level} — per-scene summary")
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])
