"""Kaplan-Meier survival analyses for per-clip and time-to-mastery framings.

Two framings, selected by ``config.analysis.survival.event``:

- **death** (default). For each scene, every clip contributes a (duration, event)
  pair: duration = ``EndFrame − StartFrame`` (frames within the clip), event = 1 if
  ``Outcome == death`` else 0 (right-censored at clip end). Per-scene KM curves
  describe how within-clip mortality risk distributes over the traversal.
- **first_clear**. For each (Subject, SceneID), find the first clip in ClipCode
  order whose outcome is ``completed``; duration = the ordinal index of that
  attempt; event = 1 if a clear was ever observed (else right-censored at the
  last attempt). KM curves describe how many attempts it takes to succeed.

Outputs:
- ``per_scene_km.csv`` — long-form KM survival table (scene_id, t, S(t), n_at_risk)
- ``per_scene_summary.csv`` — median survival + n + event counts per scene
- ``figures/`` — one PNG per scene, plus a multi-scene grid for the level
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter

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
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    s_cfg = cfg["analysis"]["survival"]
    event_mode = s_cfg["event"]
    min_attempts = int(s_cfg["min_attempts_per_scene"])

    pairs = _make_duration_event(clips, event_mode)
    pairs = _filter_min_attempts(pairs, min_attempts)
    if pairs.empty:
        log.warning("survival: no scenes meet min_attempts_per_scene=%d", min_attempts)
        return {}

    km_long, summary = _fit_km_per_scene(pairs)

    long_path = out_dir / "per_scene_km.csv"
    km_long.to_csv(long_path, index=False)
    provenance.write_sidecar(long_path, parameters=parameters, inputs=inputs)

    summary_path = out_dir / "per_scene_summary.csv"
    summary.to_csv(summary_path, index=False)
    provenance.write_sidecar(summary_path, parameters=parameters, inputs=inputs)

    paths = {"km_long": long_path, "summary": summary_path}
    for level, level_df in pairs.groupby("Level"):
        fig_path = figs_dir / f"{level}_km.png"
        _figure_level(level, level_df, fig_path, cfg=cfg)
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        paths[f"figure_{level}"] = fig_path

    log.info("survival[%s]: %d scenes, %d events", event_mode, summary["SceneID"].nunique(), int(summary["n_events"].sum()))
    return paths


def _make_duration_event(clips: pd.DataFrame, event_mode: str) -> pd.DataFrame:
    """Return one (Subject, SceneID, Level, duration, event) row per observation."""
    if event_mode == "death":
        frames = clips["EndFrame"].astype(int) - clips["StartFrame"].astype(int)
        return pd.DataFrame({
            "Subject": clips["Subject"],
            "SceneID": clips["SceneID"],
            "Level": clips["Level"],
            "duration": frames.clip(lower=1).astype(float),
            "event": (clips["Outcome"] == "death").astype(int),
        })
    if event_mode == "first_clear":
        rows = []
        for (subject, scene), g in clips.sort_values("ClipCode").groupby(["Subject", "SceneID"]):
            level = g["Level"].iloc[0]
            attempts = g.reset_index(drop=True)
            ix = attempts.index[attempts["Cleared"] == 1]
            if len(ix):
                duration = int(ix[0]) + 1
                event = 1
            else:
                duration = len(attempts)
                event = 0
            rows.append({"Subject": subject, "SceneID": scene, "Level": level,
                         "duration": float(duration), "event": event})
        return pd.DataFrame(rows)
    raise ValueError(f"Unknown event mode: {event_mode}")


def _filter_min_attempts(pairs: pd.DataFrame, min_attempts: int) -> pd.DataFrame:
    counts = pairs.groupby("SceneID").size()
    keep = counts[counts >= min_attempts].index
    return pairs[pairs["SceneID"].isin(keep)].copy()


def _fit_km_per_scene(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    summary_rows = []
    for scene, g in pairs.groupby("SceneID"):
        kmf = KaplanMeierFitter()
        kmf.fit(g["duration"], event_observed=g["event"])
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["t", "S"]
        sf["SceneID"] = scene
        sf["n_at_risk"] = kmf.event_table["at_risk"].to_numpy()
        long_rows.append(sf[["SceneID", "t", "S", "n_at_risk"]])

        median = kmf.median_survival_time_
        summary_rows.append({
            "SceneID": scene,
            "Level": g["Level"].iloc[0],
            "n_observations": len(g),
            "n_events": int(g["event"].sum()),
            "median_survival": median if not pd.isna(median) else None,
        })
    return pd.concat(long_rows, ignore_index=True), pd.DataFrame(summary_rows)


def _figure_level(level: str, level_df: pd.DataFrame, out_path: Path, *, cfg: dict) -> None:
    style = plots.style(cfg)
    scenes = sorted(level_df["SceneID"].unique(), key=lambda s: int(s.split("s")[-1]))
    n = len(scenes)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows), squeeze=False, sharey=True)
    for i, scene in enumerate(scenes):
        ax = axes[i // cols][i % cols]
        g = level_df[level_df["SceneID"] == scene]
        kmf = KaplanMeierFitter()
        kmf.fit(g["duration"], event_observed=g["event"], label=scene)
        kmf.plot_survival_function(ax=ax, ci_show=True)
        ax.set_title(scene, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
        ax.legend([], frameon=False)
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis("off")
    fig.suptitle(f"{level} — KM survival per scene")
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])
