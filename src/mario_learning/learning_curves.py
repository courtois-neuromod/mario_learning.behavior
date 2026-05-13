"""Moving-average learning curves.

Per-subject views are primary (per [[feedback-no-subject-averaging]]). The
**primary** axis is per-pattern (per [[feedback-patterns-over-scenes]]) — for
each (Subject, Pattern, Variable) triplet, the clips touching that pattern are
sorted by ClipCode and a centered rolling mean is applied to the variable.

A secondary per-level view is kept as a complementary diagnostic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
    lc_cfg = cfg["analysis"]["learning_curves"]
    window = int(lc_cfg["smoothing_window"])
    variables = list(lc_cfg["variables"])

    missing = [v for v in variables if v not in clips.columns]
    if missing:
        raise KeyError(f"Variables not in clips DataFrame: {missing}")

    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    paths: dict[str, Path] = {}

    # Primary: per-pattern curves
    long_df = utils.attach_patterns(clips)
    pattern_curves = _smooth(long_df, ["Subject", "pattern"], variables, window)
    by_pattern_path = out_dir / "learning_curves_by_pattern.csv"
    pattern_curves.to_csv(by_pattern_path, index=False)
    provenance.write_sidecar(by_pattern_path, parameters=parameters, inputs=inputs)
    paths["table_patterns"] = by_pattern_path
    for subject, sub_df in pattern_curves.groupby("Subject"):
        fig_path = figs_dir / f"sub-{subject}_learning_curves_patterns.png"
        _figure_patterns(sub_df, variables, window, fig_path, cfg=cfg)
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        paths[f"figure_patterns_sub-{subject}"] = fig_path

    # Secondary: per-level curves (kept as a complementary view)
    level_curves = _smooth(clips, ["Subject", "Level"], variables, window)
    by_level_path = out_dir / "learning_curves_by_level.csv"
    level_curves.to_csv(by_level_path, index=False)
    provenance.write_sidecar(by_level_path, parameters=parameters, inputs=inputs)
    paths["table_levels"] = by_level_path
    for subject, sub_df in level_curves.groupby("Subject"):
        fig_path = figs_dir / f"sub-{subject}_learning_curves_levels.png"
        _figure_levels(sub_df, variables, window, fig_path, cfg=cfg)
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        paths[f"figure_levels_sub-{subject}"] = fig_path

    log.info(
        "learning_curves: %d subjects, %d patterns, %d levels, window=%d, vars=%s",
        pattern_curves["Subject"].nunique(),
        pattern_curves["pattern"].nunique(),
        level_curves["Level"].nunique(),
        window, variables,
    )
    return paths


def _smooth(
    df: pd.DataFrame,
    group_keys: list[str],
    variables: list[str],
    window: int,
) -> pd.DataFrame:
    """For each `group_keys` group, sort by ClipCode and rolling-mean each variable.

    The output adds a `clip_index` (within-group rank in clip-code order) for
    plotting on a normalized x-axis.
    """
    pieces = []
    keep_cols = [*group_keys, "Phase", "ClipCode"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    for _key, g in df.groupby(group_keys):
        g = g.sort_values("ClipCode").reset_index(drop=True)
        out = g[keep_cols].copy()
        out["clip_index"] = range(len(out))
        for var in variables:
            series = pd.to_numeric(g[var], errors="coerce")
            out[var] = series.to_numpy()
            out[f"{var}_smoothed"] = plots.moving_average(series, window)
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def _figure_patterns(
    sub_df: pd.DataFrame,
    variables: list[str],
    window: int,
    out_path: Path,
    *,
    cfg: dict,
) -> None:
    style = plots.style(cfg)
    patterns = sorted(sub_df["pattern"].unique())
    n_pat = len(patterns)
    n_vars = len(variables)
    fig, axes = plt.subplots(
        n_vars, n_pat,
        figsize=(2.0 * n_pat, 2.4 * n_vars),
        sharex="col", sharey="row",
        squeeze=False,
    )
    for j, pattern in enumerate(patterns):
        sub = sub_df[sub_df["pattern"] == pattern]
        for i, var in enumerate(variables):
            ax = axes[i, j]
            ax.plot(sub["clip_index"], sub[var], color="lightgray", lw=0.5)
            ax.plot(sub["clip_index"], sub[f"{var}_smoothed"], color="C0", lw=1.5)
            if i == 0:
                ax.set_title(pattern, fontsize=7, rotation=0)
            if j == 0:
                ax.set_ylabel(var, fontsize=9)
            if i == n_vars - 1:
                ax.set_xlabel("clip idx", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"sub-{sub_df['Subject'].iloc[0]} — learning curves by pattern (window={window})")
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])


def _figure_levels(
    sub_df: pd.DataFrame,
    variables: list[str],
    window: int,
    out_path: Path,
    *,
    cfg: dict,
) -> None:
    style = plots.style(cfg)
    levels = sorted(sub_df["Level"].unique())
    n_levels = len(levels)
    n_vars = len(variables)
    fig, axes = plt.subplots(
        n_vars, n_levels,
        figsize=(2.4 * n_levels, 2.4 * n_vars),
        sharex="col", sharey="row",
        squeeze=False,
    )
    for j, level in enumerate(levels):
        sub = sub_df[sub_df["Level"] == level]
        for i, var in enumerate(variables):
            ax = axes[i, j]
            ax.plot(sub["clip_index"], sub[var], color="lightgray", lw=0.5)
            ax.plot(sub["clip_index"], sub[f"{var}_smoothed"], color="C1", lw=1.5)
            if i == 0:
                ax.set_title(level, fontsize=8)
            if j == 0:
                ax.set_ylabel(var, fontsize=9)
            if i == n_vars - 1:
                ax.set_xlabel("clip idx", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"sub-{sub_df['Subject'].iloc[0]} — learning curves by level (secondary, window={window})")
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])
