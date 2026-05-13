"""Per-pattern performance trajectories and aggregates per subject.

For each (Subject, Pattern) pair, aggregate the clips touching that pattern:
mean Cleared / Duration / Hits / X_Traveled. Per-(Subject, Pattern) is the
primary cell, in line with [[feedback-no-subject-averaging]] and
[[feedback-patterns-over-scenes]].

Figures: for each subject, one panel per pattern showing clip-order
trajectories of Cleared (color) with Duration (size) — the per-pattern
analogue of the older per-scene panel grid.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mario_learning import plots, provenance, utils

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

    long_df = utils.attach_patterns(clips)
    per_pattern = _per_pattern_table(long_df)
    table_path = out_dir / "per_subject_pattern.csv"
    per_pattern.to_csv(table_path, index=False)
    provenance.write_sidecar(table_path, parameters=parameters, inputs=inputs)
    paths: dict[str, Path] = {"table": table_path}

    for subject, sub_long in long_df.groupby("Subject"):
        fig_path = figs_dir / f"sub-{subject}_pattern_perf.png"
        _figure_subject(subject, sub_long, fig_path, cfg=cfg)
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        paths[f"figure_sub-{subject}"] = fig_path

    log.info("pattern_performance: %d subjects × %d patterns",
             per_pattern["Subject"].nunique(), per_pattern["pattern"].nunique())
    return paths


def _per_pattern_table(long_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in METRICS if m in long_df.columns]
    g = long_df.groupby(["Subject", "pattern"])
    table = g[metrics].mean(numeric_only=True).reset_index()
    table["n_clips"] = g.size().values
    table["n_scenes"] = g["SceneID"].nunique().values
    return table.sort_values(["Subject", "pattern"])


def _figure_subject(subject: str, sub_long: pd.DataFrame, out_path: Path, *, cfg: dict) -> None:
    style = plots.style(cfg)
    sub_long = sub_long.sort_values("ClipCode").copy()
    sub_long["rep_idx"] = sub_long.groupby("pattern").cumcount()
    patterns = sorted(sub_long["pattern"].unique())
    n = len(patterns)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.0 * rows), squeeze=False, sharey=True)
    for i, pattern in enumerate(patterns):
        ax = axes[i // cols][i % cols]
        pat = sub_long[sub_long["pattern"] == pattern]
        cleared = pat["Cleared"].to_numpy()
        rep = pat["rep_idx"].to_numpy()
        durations = pat["Duration"].fillna(0).to_numpy()
        ax.scatter(
            rep, cleared,
            s=15 + 8 * durations,
            c=cleared, cmap="RdYlGn",
            vmin=0, vmax=1,
            edgecolor="k", lw=0.3, alpha=0.7,
        )
        ax.set_title(f"{pattern} (n={len(pat)})", fontsize=8)
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["death", "clear"], fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis("off")
    fig.suptitle(f"sub-{subject} — per-pattern trajectories (size ∝ duration)")
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])
