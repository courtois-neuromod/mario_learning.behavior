"""Performance by scene pattern, plus a per-pattern learning metric.

Joins each clip to the 27 binary scene-feature columns from Zenodo, then for
each pattern computes:

- Mean ``Cleared`` overall and per phase (discovery / practice).
- A learning metric = ``early_stage_clear × (late_stage_clear − early_stage_clear)``
  with the early/late split taken at the per-subject median ClipCode within each
  phase.

Outputs a long-form CSV plus per-subject bar plots of clear-rate by pattern.
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
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    patterns = utils.scene_pattern_matrix()
    clips_long = _attach_patterns(clips, patterns)

    pd_cfg = cfg["analysis"]["pattern_difficulty"]
    phases = list(pd_cfg["phases"])
    split_mode = pd_cfg.get("early_late_split", "median")

    metrics = _compute_metrics(clips_long, phases=phases, split_mode=split_mode)
    metrics_path = out_dir / "pattern_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    provenance.write_sidecar(metrics_path, parameters=parameters, inputs=inputs)
    paths: dict[str, Path] = {"table": metrics_path}

    for subject, sub_df in metrics.groupby("Subject"):
        fig_path = figs_dir / f"sub-{subject}_cleared_by_pattern.png"
        _figure(sub_df, subject, fig_path, cfg=cfg)
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        paths[f"figure_sub-{subject}"] = fig_path

    log.info("pattern_difficulty: %d patterns × %d subjects", metrics["pattern"].nunique(), metrics["Subject"].nunique())
    return paths


def _attach_patterns(clips: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    """Long-form: one row per (clip, pattern) where the pattern flag is 1."""
    patterns = patterns.reset_index().melt(
        id_vars="scene_id", var_name="pattern", value_name="present"
    )
    patterns = patterns[patterns["present"] == 1].drop(columns="present")
    merged = clips.merge(patterns, left_on="SceneID", right_on="scene_id", how="inner")
    return merged.drop(columns="scene_id")


def _compute_metrics(
    long_df: pd.DataFrame,
    *,
    phases: list[str],
    split_mode: str,
) -> pd.DataFrame:
    rows = []
    for (subject, pattern), g in long_df.groupby(["Subject", "pattern"]):
        row: dict = {"Subject": subject, "pattern": pattern, "n_clips": len(g)}
        for phase in phases:
            ph = g[g["Phase"] == phase]
            row[f"clear_{phase}"] = ph["Cleared"].mean() if not ph.empty else float("nan")
            if split_mode == "median" and len(ph) >= 4:
                codes = ph["ClipCode"].astype("int64")
                median_code = codes.median()
                early = ph[codes <= median_code]
                late = ph[codes > median_code]
                row[f"clear_early_{phase}"] = early["Cleared"].mean() if not early.empty else float("nan")
                row[f"clear_late_{phase}"] = late["Cleared"].mean() if not late.empty else float("nan")
        # Learning metric: anchored on discovery if available, else first listed phase.
        anchor = "discovery" if "discovery" in phases else phases[0]
        early = row.get(f"clear_early_{anchor}")
        late = row.get(f"clear_late_{anchor}")
        if early is not None and late is not None and pd.notna(early) and pd.notna(late):
            row["learning_metric"] = early * (late - early)
        else:
            row["learning_metric"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Subject", "pattern"])


def _figure(sub_df: pd.DataFrame, subject: str, out_path: Path, *, cfg: dict) -> None:
    style = plots.style(cfg)
    sub_df = sub_df.sort_values("clear_discovery" if "clear_discovery" in sub_df.columns else "Subject")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(sub_df))))
    patterns = sub_df["pattern"].tolist()
    y = range(len(patterns))
    ax.barh(y, sub_df["clear_discovery"], color="steelblue", label="discovery")
    if "clear_practice" in sub_df.columns:
        ax.barh(y, sub_df["clear_practice"], color="darkorange", alpha=0.6, label="practice")
    ax.set_yticks(list(y))
    ax.set_yticklabels(patterns, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Clear rate")
    ax.set_title(f"sub-{subject} — clear rate by scene pattern")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])
