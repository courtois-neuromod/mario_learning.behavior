"""Per-subject pattern grids of performance variables.

For each subject, emit a grid figure with patterns as columns and
variables (Cleared, Duration, Hits_taken) as rows. Each cell shows the
variable's mean for that pattern, split by Stage (early/late × discovery/practice
— see [[feedback-four-stage-split]]). Pattern is the analytical unit
(see [[feedback-patterns-over-scenes]]).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mario_learning import plots, provenance, utils

log = logging.getLogger(__name__)

VARIABLES = ["Cleared", "Duration", "Hits_taken"]
VARIABLE_DIRECTION = {"Cleared": "up", "Duration": "down", "Hits_taken": "down"}


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

    staged = utils.add_stage_column(clips)
    long_df = utils.attach_patterns(staged)
    per_pattern = _aggregate(long_df)
    csv_path = out_dir / "per_subject_pattern_stage.csv"
    per_pattern.to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, parameters=parameters, inputs=inputs)

    paths: dict[str, Path] = {"table": csv_path}
    for subject, sub_df in per_pattern.groupby("Subject"):
        fig_path = figs_dir / f"sub-{subject}_pattern_summary.png"
        _figure_for_subject(subject, sub_df, fig_path, cfg=cfg)
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        paths[f"figure_sub-{subject}"] = fig_path

    log.info("summary: %d subjects × %d patterns", per_pattern["Subject"].nunique(), per_pattern["pattern"].nunique())
    return paths


def _aggregate(long_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in VARIABLES if c in long_df.columns]
    grouped = (
        long_df.groupby(["Subject", "pattern", "Stage"], observed=True)[cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped["Stage"] = pd.Categorical(grouped["Stage"], categories=utils.STAGES, ordered=True)
    return grouped.sort_values(["Subject", "pattern", "Stage"])


def _figure_for_subject(subject: str, sub_df: pd.DataFrame, out_path: Path, *, cfg: dict) -> None:
    style = plots.style(cfg)
    patterns = sorted(sub_df["pattern"].unique())
    n_pat = len(patterns)
    n_vars = len(VARIABLES)
    fig, axes = plt.subplots(
        n_vars, 1,
        figsize=(max(10, 0.55 * n_pat + 2), 2.5 * n_vars),
        sharex=True, squeeze=False,
    )
    palette = plt.get_cmap("viridis")
    stage_colors = {s: palette(i / (len(utils.STAGES) - 1)) for i, s in enumerate(utils.STAGES)}
    width = 0.13
    group_spacing = 1.4  # gap between pattern groups: 1.4 - 6*0.13 = 0.62
    x = np.arange(n_pat) * group_spacing
    for i, var in enumerate(VARIABLES):
        ax = axes[i][0]
        for k, stage in enumerate(utils.STAGES):
            sub = sub_df[sub_df["Stage"] == stage].set_index("pattern").reindex(patterns)
            ax.bar(x + k * width, sub[var].fillna(0).to_numpy(), width,
                   color=stage_colors[stage], label=stage if i == 0 else None)
        ax.set_ylabel(var)
        plots.add_direction_arrow(ax, VARIABLE_DIRECTION[var])
        ax.grid(axis="y", alpha=0.3)
    axes[-1][0].set_xticks(x + width * 2.5)
    axes[-1][0].set_xticklabels(patterns, rotation=45, ha="right", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(utils.STAGES), fontsize=7, frameon=True)
    fig.suptitle(f"sub-{subject} — performance by pattern × stage", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plots.save_figure(fig, out_path, dpi=style["dpi"])
