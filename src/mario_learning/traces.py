"""Player x/y position traces overlaid on level/scene background images.

For every clip in the filtered DataFrame, read the companion ``_variables.json``,
reconstruct the in-level x coordinate (``player_x_posHi * 256 + player_x_posLo``)
and y coordinate (``player_y_pos``), and draw the trace on:

- the full-level background (one image per ``(Subject, Level)``)
- the per-scene background (one image per ``(Subject, SceneID)``)

Completed clips are drawn green, deaths red. Scene start/end x positions are
read from the Zenodo ``scenes_mastersheet.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from mario_learning import loader, plots, provenance, utils

log = logging.getLogger(__name__)


def run(
    clips: pd.DataFrame,
    out_dir: Path,
    *,
    parameters: dict,
    inputs: list,
    cfg: dict,
    dataset_path: Path,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_cfg = cfg["analysis"]["traces"]
    bg_root = Path(t_cfg["background_source"]).expanduser()
    if not bg_root.exists():
        raise FileNotFoundError(
            f"traces.background_source = {bg_root} does not exist. "
            "Point it at a directory containing level_backgrounds/ and scene_backgrounds/."
        )

    scene_starts = _scene_start_map()
    clip_limit = t_cfg.get("clip_limit")

    paths: dict[str, Path] = {}
    # Per (subject, level) overlays
    for (subject, level), g in clips.groupby(["Subject", "Level"]):
        bg = bg_root / "level_backgrounds" / f"{level}.png"
        if not bg.exists():
            log.warning("Level background missing for %s: %s", level, bg)
            continue
        sub_dir = out_dir / f"sub-{subject}" / "levels"
        sub_dir.mkdir(parents=True, exist_ok=True)
        out_path = sub_dir / f"{level}.png"
        sample = _sample_clips(g, clip_limit)
        _figure_level(sample, bg, level, out_path, cfg=cfg, dataset_path=dataset_path)
        provenance.write_sidecar(out_path, parameters=parameters, inputs=inputs)
        paths[f"level_sub-{subject}_{level}"] = out_path

    # Per (subject, scene) overlays
    for (subject, scene_id), g in clips.groupby(["Subject", "SceneID"]):
        bg = bg_root / "scene_backgrounds" / f"{scene_id}.png"
        if not bg.exists():
            continue
        if scene_id not in scene_starts:
            continue
        sub_dir = out_dir / f"sub-{subject}" / "scenes"
        sub_dir.mkdir(parents=True, exist_ok=True)
        out_path = sub_dir / f"{scene_id}.png"
        sample = _sample_clips(g, clip_limit)
        _figure_scene(sample, bg, scene_id, scene_starts[scene_id], out_path, cfg=cfg, dataset_path=dataset_path)
        provenance.write_sidecar(out_path, parameters=parameters, inputs=inputs)
        paths[f"scene_sub-{subject}_{scene_id}"] = out_path

    log.info("traces: %d (subject, level) figures, %d (subject, scene) figures",
             sum(1 for k in paths if k.startswith("level_")),
             sum(1 for k in paths if k.startswith("scene_")))
    return paths


def _scene_start_map() -> dict[str, int]:
    df = utils.load_scene_annotations()
    return dict(zip(df["scene_id"], df["Entry point"].astype(int), strict=False))


def _sample_clips(g: pd.DataFrame, clip_limit: int | None) -> pd.DataFrame:
    if clip_limit and len(g) > clip_limit:
        # Spread evenly across the clip-code order so we sample the whole trajectory.
        idx = np.linspace(0, len(g) - 1, clip_limit).astype(int)
        return g.iloc[idx]
    return g


def _player_xy(variables: dict) -> tuple[np.ndarray, np.ndarray]:
    xhi = np.asarray(variables["player_x_posHi"])
    xlo = np.asarray(variables["player_x_posLo"])
    y = np.asarray(variables["player_y_pos"])
    return xhi * 256 + xlo, y


def _clip_trace(dataset_path: Path, summary_relpath: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        v = loader.load_variables(dataset_path, summary_relpath)
    except (FileNotFoundError, OSError, ValueError) as e:
        log.debug("variables missing for %s: %s", summary_relpath, e)
        return None
    x_all, y_all = _player_xy(v)
    if start >= len(x_all) or end > len(x_all):
        return x_all, y_all
    return x_all[start:end], y_all[start:end]


def _figure_level(
    clips: pd.DataFrame,
    bg_path: Path,
    level: str,
    out_path: Path,
    *,
    cfg: dict,
    dataset_path: Path,
) -> None:
    style = plots.style(cfg)
    bg = np.asarray(Image.open(bg_path).convert("RGB"))
    fig, ax = plt.subplots(figsize=(min(20, bg.shape[1] / 80), bg.shape[0] / 80))
    ax.imshow(bg)
    for _, row in clips.iterrows():
        trace = _clip_trace(dataset_path, row["summary_path"], int(row["StartFrame"]), int(row["EndFrame"]))
        if trace is None:
            continue
        x, y = trace
        color = "#2ca02c" if row["Cleared"] == 1 else "#d62728"
        ax.plot(x, y, color=color, lw=0.6, alpha=0.45)
    ax.set_xlim(0, bg.shape[1])
    ax.set_ylim(bg.shape[0], 0)
    ax.set_axis_off()
    ax.set_title(f"sub-{clips['Subject'].iloc[0]} — {level} ({len(clips)} clips)")
    fig.tight_layout(pad=0)
    plots.save_figure(fig, out_path, dpi=style["dpi"])


def _figure_scene(
    clips: pd.DataFrame,
    bg_path: Path,
    scene_id: str,
    scene_start_x: int,
    out_path: Path,
    *,
    cfg: dict,
    dataset_path: Path,
) -> None:
    style = plots.style(cfg)
    bg = np.asarray(Image.open(bg_path).convert("RGB"))
    fig, ax = plt.subplots(figsize=(min(10, bg.shape[1] / 40), bg.shape[0] / 40))
    ax.imshow(bg)
    for _, row in clips.iterrows():
        trace = _clip_trace(dataset_path, row["summary_path"], int(row["StartFrame"]), int(row["EndFrame"]))
        if trace is None:
            continue
        x, y = trace
        color = "#2ca02c" if row["Cleared"] == 1 else "#d62728"
        ax.plot(x - scene_start_x, y, color=color, lw=0.8, alpha=0.55)
    ax.set_xlim(0, bg.shape[1])
    ax.set_ylim(bg.shape[0], 0)
    ax.set_axis_off()
    ax.set_title(f"sub-{clips['Subject'].iloc[0]} — {scene_id} ({len(clips)} clips)")
    fig.tight_layout(pad=0)
    plots.save_figure(fig, out_path, dpi=style["dpi"])
