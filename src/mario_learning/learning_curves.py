"""Moving-average learning curves over the clip-code time axis.

For each (Subject, Level, Variable) triplet, sort clips by ClipCode and apply
a centered rolling mean (window from config) to the variable. Emits one CSV
per dataset with the smoothed series plus per-subject figures (one panel per
level, one line per variable).
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lc_cfg = cfg["analysis"]["learning_curves"]
    window = int(lc_cfg["smoothing_window"])
    variables = list(lc_cfg["variables"])

    missing = [v for v in variables if v not in clips.columns]
    if missing:
        raise KeyError(f"Variables not in clips DataFrame: {missing}")

    smoothed = _smooth(clips, variables, window)

    paths: dict[str, Path] = {}
    csv_path = out_dir / "learning_curves.csv"
    smoothed.to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, parameters=parameters, inputs=inputs)
    paths["table"] = csv_path

    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    for subject, sub_df in smoothed.groupby("Subject"):
        fig_path = figs_dir / f"sub-{subject}_learning_curves.png"
        _figure(sub_df, variables, window, fig_path, cfg=cfg)
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        paths[f"figure_sub-{subject}"] = fig_path

    log.info("learning_curves: %d subjects, window=%d, vars=%s", smoothed["Subject"].nunique(), window, variables)
    return paths


def _smooth(clips: pd.DataFrame, variables: list[str], window: int) -> pd.DataFrame:
    """Per (Subject, Level), sort by ClipCode and rolling-mean each variable."""
    pieces = []
    for _key, g in clips.groupby(["Subject", "Level"]):
        g = g.sort_values("ClipCode").reset_index(drop=True)
        out = g[["Subject", "Level", "Phase", "ClipCode"]].copy()
        out["clip_index"] = range(len(out))
        for var in variables:
            series = pd.to_numeric(g[var], errors="coerce")
            out[var] = series.to_numpy()
            out[f"{var}_smoothed"] = plots.moving_average(series, window)
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def _figure(sub_df: pd.DataFrame, variables: list[str], window: int, out_path: Path, *, cfg: dict) -> None:
    style = plots.style(cfg)
    levels = sorted(sub_df["Level"].unique())
    n_levels = len(levels)
    n_vars = len(variables)
    fig, axes = plt.subplots(
        n_vars, n_levels,
        figsize=(3.2 * n_levels, 2.4 * n_vars),
        sharex="col",
        squeeze=False,
    )
    for j, level in enumerate(levels):
        level_df = sub_df[sub_df["Level"] == level]
        for i, var in enumerate(variables):
            ax = axes[i, j]
            ax.plot(level_df["clip_index"], level_df[var], color="lightgray", lw=0.5)
            ax.plot(level_df["clip_index"], level_df[f"{var}_smoothed"], color="C0", lw=1.5)
            if i == 0:
                ax.set_title(level, fontsize=9)
            if j == 0:
                ax.set_ylabel(var, fontsize=9)
            if i == n_vars - 1:
                ax.set_xlabel("clip index")
    fig.suptitle(f"sub-{sub_df['Subject'].iloc[0]} — learning curves (window={window})")
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])
