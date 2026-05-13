"""Kaplan-Meier survival analyses, per-pattern (primary) and per-scene (secondary).

Two framings, selected by ``config.analysis.survival.event``:

- **death** (default). Every clip contributes (duration, event) where
  duration = ``EndFrame − StartFrame`` and event = 1 iff ``Outcome == death``
  (right-censored at clip end). KM curves describe how within-clip mortality
  risk distributes over the traversal.
- **first_clear**. For each (Subject, group) the first clip in ClipCode order
  whose outcome is `completed` defines the duration (ordinal index of that
  attempt); event = 1 iff a clear was ever observed. Curves describe time-to-
  mastery.

Primary grain is per (Subject, Pattern) — see [[feedback-patterns-over-scenes]]
and [[feedback-no-subject-averaging]]. Per-scene KM (the previous primary) is
retained as a per-level diagnostic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter

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

    s_cfg = cfg["analysis"]["survival"]
    event_mode = s_cfg["event"]
    min_attempts = int(s_cfg["min_attempts_per_scene"])
    paths: dict[str, Path] = {}

    # Primary: per (Subject, Pattern)
    long_df = utils.attach_patterns(clips)
    pattern_pairs = _make_duration_event(long_df, event_mode, group_keys=["Subject", "pattern"])
    pattern_pairs = _filter_min(pattern_pairs, "pattern", min_attempts)
    if not pattern_pairs.empty:
        pkm_long, pkm_summary = _fit_km(pattern_pairs, group_keys=["Subject", "pattern"])
        plong = out_dir / "per_subject_pattern_km.csv"
        pkm_long.to_csv(plong, index=False)
        provenance.write_sidecar(plong, parameters=parameters, inputs=inputs)
        psum = out_dir / "per_subject_pattern_summary.csv"
        pkm_summary.to_csv(psum, index=False)
        provenance.write_sidecar(psum, parameters=parameters, inputs=inputs)
        paths["pattern_km_long"] = plong
        paths["pattern_summary"] = psum
        for subject, sub_df in pattern_pairs.groupby("Subject"):
            fig_path = figs_dir / f"sub-{subject}_pattern_km.png"
            _figure_grid(sub_df, group_col="pattern", title=f"sub-{subject} — KM by pattern",
                         out_path=fig_path, cfg=cfg)
            provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
            paths[f"figure_pattern_sub-{subject}"] = fig_path
        log.info("survival[%s] (primary, per-pattern): %d subjects × %d patterns",
                 event_mode, pattern_pairs["Subject"].nunique(), pattern_pairs["pattern"].nunique())

    # Secondary: per-scene KM (kept for diagnostics)
    scene_pairs = _make_duration_event(clips, event_mode, group_keys=["Subject", "SceneID"])
    scene_pairs = _filter_min(scene_pairs, "SceneID", min_attempts)
    if not scene_pairs.empty:
        skm_long, skm_summary = _fit_km(scene_pairs, group_keys=["SceneID"])
        slong = out_dir / "per_scene_km.csv"
        skm_long.to_csv(slong, index=False)
        provenance.write_sidecar(slong, parameters=parameters, inputs=inputs)
        ssum = out_dir / "per_scene_summary.csv"
        skm_summary.to_csv(ssum, index=False)
        provenance.write_sidecar(ssum, parameters=parameters, inputs=inputs)
        paths["scene_km_long"] = slong
        paths["scene_summary"] = ssum
        for level, level_df in scene_pairs.groupby("Level"):
            fig_path = figs_dir / f"{level}_km.png"
            _figure_grid(level_df, group_col="SceneID", title=f"{level} — KM per scene",
                         out_path=fig_path, cfg=cfg)
            provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
            paths[f"figure_scene_{level}"] = fig_path
        log.info("survival[%s] (secondary, per-scene): %d scenes, %d events",
                 event_mode, skm_summary["SceneID"].nunique(), int(skm_summary["n_events"].sum()))
    return paths


def _make_duration_event(
    df: pd.DataFrame,
    event_mode: str,
    *,
    group_keys: list[str],
) -> pd.DataFrame:
    """Return a DataFrame with columns: *group_keys, Level, duration, event."""
    if event_mode == "death":
        out = pd.DataFrame({
            "duration": (df["EndFrame"].astype(int) - df["StartFrame"].astype(int)).clip(lower=1).astype(float),
            "event": (df["Outcome"] == "death").astype(int),
            "Level": df["Level"],
        })
        for key in group_keys:
            out[key] = df[key].values
        return out
    if event_mode == "first_clear":
        rows = []
        for key_vals, g in df.sort_values("ClipCode").groupby(group_keys):
            level = g["Level"].iloc[0]
            attempts = g.reset_index(drop=True)
            ix = attempts.index[attempts["Cleared"] == 1]
            duration = (int(ix[0]) + 1) if len(ix) else len(attempts)
            event = 1 if len(ix) else 0
            row = dict(zip(group_keys, key_vals if isinstance(key_vals, tuple) else (key_vals,), strict=False))
            row.update({"Level": level, "duration": float(duration), "event": event})
            rows.append(row)
        return pd.DataFrame(rows)
    raise ValueError(f"Unknown event mode: {event_mode}")


def _filter_min(pairs: pd.DataFrame, key: str, min_attempts: int) -> pd.DataFrame:
    counts = pairs.groupby(key).size()
    keep = counts[counts >= min_attempts].index
    return pairs[pairs[key].isin(keep)].copy()


def _fit_km(pairs: pd.DataFrame, *, group_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    summary_rows = []
    for key_vals, g in pairs.groupby(group_keys):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        kmf = KaplanMeierFitter()
        kmf.fit(g["duration"], event_observed=g["event"])
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["t", "S"]
        for k, v in zip(group_keys, key_vals, strict=False):
            sf[k] = v
        sf["n_at_risk"] = kmf.event_table["at_risk"].to_numpy()
        long_rows.append(sf)

        median = kmf.median_survival_time_
        row = dict(zip(group_keys, key_vals, strict=False))
        row.update({
            "Level": g["Level"].iloc[0] if "Level" in g.columns else None,
            "n_observations": len(g),
            "n_events": int(g["event"].sum()),
            "median_survival": median if not pd.isna(median) else None,
        })
        summary_rows.append(row)
    return pd.concat(long_rows, ignore_index=True), pd.DataFrame(summary_rows)


def _figure_grid(
    pairs: pd.DataFrame,
    *,
    group_col: str,
    title: str,
    out_path: Path,
    cfg: dict,
) -> None:
    style = plots.style(cfg)
    groups = sorted(pairs[group_col].unique())
    n = len(groups)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows), squeeze=False, sharey=True)
    for i, group in enumerate(groups):
        ax = axes[i // cols][i % cols]
        g = pairs[pairs[group_col] == group]
        kmf = KaplanMeierFitter()
        kmf.fit(g["duration"], event_observed=g["event"], label=str(group))
        kmf.plot_survival_function(ax=ax, ci_show=True)
        ax.set_title(f"{group}", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
        ax.legend([], frameon=False)
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])
