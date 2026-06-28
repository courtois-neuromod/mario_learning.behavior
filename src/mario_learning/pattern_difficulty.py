"""Per-pattern clear rates across the 6-stage discovery/practice split.

For each (Subject, Pattern, Stage) computes:

- clear rate (mean Cleared)
- median duration, median hits
- a stage-pair learning metric: (late_practice − early_discovery) per pattern

Per [[feedback-no-subject-averaging]], every metric is reported per subject.
Stage ∈ {early_discovery, middle_discovery, late_discovery,
early_practice, middle_practice, late_practice}. Per [[feedback-patterns-over-scenes]],
pattern is the analytical unit.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mario_learning import plots, provenance, utils

log = logging.getLogger(__name__)


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

    metrics = _compute_metrics(long_df)
    metrics_path = out_dir / "pattern_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    provenance.write_sidecar(metrics_path, parameters=parameters, inputs=inputs)
    paths: dict[str, Path] = {"table": metrics_path}

    for subject, sub_df in metrics.groupby("Subject"):
        fig_path = figs_dir / f"sub-{subject}_cleared_by_pattern.png"
        _figure_subject(sub_df, subject, fig_path, cfg=cfg)
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        paths[f"figure_sub-{subject}"] = fig_path

    # Pooled-subjects "headline" view: bars per pattern × stage (viridis).
    # One pooled figure per analysis is acceptable as a secondary headline
    # even though per-subject is the primary grain (see feedback-no-subject-averaging).
    pooled_path = figs_dir / "cleared_by_pattern_pooled_by_stage.png"
    _figure_pooled_by_stage(long_df, pooled_path, cfg=cfg)
    provenance.write_sidecar(pooled_path, parameters=parameters, inputs=inputs)
    paths["figure_pooled"] = pooled_path

    log.info("pattern_difficulty: %d patterns × %d subjects, 6 stages",
             metrics["pattern"].nunique(), metrics["Subject"].nunique())
    return paths


def _compute_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per (Subject, Pattern, Stage): n, clear rate, duration, hits.

    Pivots to a wide-form row per (Subject, Pattern) so each row records all
    six stages, plus an `improvement = late_practice − early_discovery` column.
    """
    g = long_df.groupby(["Subject", "pattern", "Stage"], observed=True)
    long = pd.DataFrame({
        "n_clips": g.size(),
        "clear_rate": g["Cleared"].mean(),
        "median_duration_s": g["Duration"].median(),
        "median_hits": g["Hits_taken"].median(),
    }).reset_index()

    clear_wide = long.pivot_table(
        index=["Subject", "pattern"], columns="Stage", values="clear_rate", observed=True
    )
    n_wide = long.pivot_table(
        index=["Subject", "pattern"], columns="Stage", values="n_clips", observed=True
    ).add_prefix("n_")

    for stage in utils.STAGES:
        if stage not in clear_wide.columns:
            clear_wide[stage] = np.nan
    clear_wide = clear_wide[utils.STAGES]
    clear_wide["improvement"] = clear_wide["late_practice"] - clear_wide["early_discovery"]

    wide = pd.concat([clear_wide, n_wide], axis=1).reset_index()
    return wide.sort_values(["Subject", "pattern"])


def _figure_pooled_by_stage(long_df: pd.DataFrame, out_path: Path, *, cfg: dict) -> None:
    """Top panel of the canonical 'cleared_by_pattern' figure: humans pooled, 6 viridis bars per pattern."""
    style = plots.style(cfg)
    grouped = (
        long_df.groupby(["pattern", "Stage"], observed=True)["Cleared"].mean().reset_index()
    )
    grouped["Stage"] = pd.Categorical(grouped["Stage"], categories=utils.STAGES, ordered=True)
    patterns = sorted(grouped["pattern"].unique())
    fig, ax = plt.subplots(figsize=(16, 4))
    sns.barplot(
        data=grouped, x="pattern", y="Cleared", hue="Stage",
        order=patterns, hue_order=utils.STAGES,
        palette="viridis", ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Pattern")
    ax.set_ylabel("Cleared")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=60, labelsize=9)
    ax.legend(title="Stage", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])


def _figure_subject(sub_df: pd.DataFrame, subject: str, out_path: Path, *, cfg: dict) -> None:
    style = plots.style(cfg)
    sub_df = sub_df.sort_values("improvement", ascending=False)
    patterns = sub_df["pattern"].tolist()
    n = len(patterns)
    y = np.arange(n)

    palette = plt.get_cmap("viridis")
    stage_colors = {s: palette(i / (len(utils.STAGES) - 1)) for i, s in enumerate(utils.STAGES)}
    height = 0.2

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * n)))
    for k, stage in enumerate(utils.STAGES):
        ax.barh(y + k * height, sub_df[stage].fillna(0).to_numpy(), height,
                color=stage_colors[stage], label=stage)
    ax.set_yticks(y + height * 2.5)
    ax.set_yticklabels(patterns, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Clear rate")
    ax.set_title(f"sub-{subject} — clear rate by pattern × stage (sorted by improvement)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])
