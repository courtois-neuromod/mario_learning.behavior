"""Build a clip-level DataFrame from a mario.scenes-formatted dataset.

The loader walks ``sub-*/ses-*/gamelogs/*_summary.json`` under the given root,
concatenates the per-clip metadata into a DataFrame, derives a binary
``Cleared`` column, and caches the result as parquet for fast reload. The
parquet has a JSON sidecar (see ``provenance.py``) recording the input
fingerprint so re-runs with unchanged inputs and matching selectors hit cache.

Trace data (per-frame x/y, actions, ...) lives in companion ``_variables.json``
files and is loaded lazily by ``load_variables()`` — it is too large to keep
in memory or to pre-cache for the full dataset.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from mario_learning import provenance, utils

log = logging.getLogger(__name__)

SUMMARY_COLUMNS = [
    "Subject", "Session", "Run", "Level", "SceneID", "ClipCode",
    "StartFrame", "EndFrame", "Duration", "Outcome", "Phase",
    "SourceBk2", "GameName",
    "ScoreGained", "CoinsGained", "Lives_lost", "Hits_taken",
    "Enemies_killed", "Powerups_collected", "Bricks_smashed", "X_Traveled",
]


def load_clips(
    *,
    dataset_path: os.PathLike | str,
    dataset_name: str,
    cache_dir: os.PathLike | str,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    runs: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return the per-clip DataFrame for one dataset, building/caching as needed.

    The cache is keyed on (dataset_name, subjects, sessions, runs). The sidecar
    digests every contributing ``_summary.json`` path/size/mtime; if any of
    them change, the cache misses and the parquet is rebuilt.
    """
    dataset_path = Path(dataset_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / "clips.parquet"

    summary_paths = utils.list_summary_jsons(dataset_path, subjects, sessions)
    if not summary_paths:
        raise FileNotFoundError(
            f"No *_summary.json files found under {dataset_path} matching "
            f"subjects={subjects}, sessions={sessions}."
        )

    parameters = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "subjects": subjects,
        "sessions": sessions,
        "runs": runs,
    }

    if not force and provenance.check_match(parquet_path, parameters=parameters, inputs=summary_paths):
        log.info("Cache hit: %s (%d clips)", parquet_path, _row_count(parquet_path))
        df = pd.read_parquet(parquet_path)
    else:
        log.info("Building clips DataFrame from %d summary files ...", len(summary_paths))
        df = _build_dataframe(summary_paths, dataset_name)
        df.to_parquet(parquet_path, index=False)
        provenance.write_sidecar(parquet_path, parameters=parameters, inputs=summary_paths)
        log.info("Wrote %s (%d clips)", parquet_path, len(df))

    # Run filter is post-build because run number lives in the JSON payload,
    # not in the *_summary.json filename's BIDS entities.
    run_values = utils.strip_bids_prefix(runs, "run-")
    if run_values:
        df = df[df["Run"].isin(run_values)].reset_index(drop=True)
    return df


def load_variables(
    dataset_path: os.PathLike | str,
    summary_relpath: str,
) -> dict:
    """Read a single clip's ``_variables.json`` (per-frame trace data).

    ``summary_relpath`` is the dataset-relative path stored in the clips
    DataFrame's index. The variables file lives next to it.
    """
    summary_path = Path(dataset_path) / summary_relpath
    variables_path = summary_path.with_name(summary_path.name.replace("_summary.json", "_variables.json"))
    with variables_path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _clip_relpath(item: dict, gamelogs_reldir: Path, suffix: str = "_summary.json") -> str:
    """Reconstruct the per-clip filename from summary metadata fields.

    Used when loading a consolidated (array) summary file to recover the
    per-clip path needed to locate companion ``_variables.json`` files.
    Returns an empty string when any required field is missing.
    """
    sub      = item.get("Subject", "")
    ses      = item.get("Session", "")
    level    = item.get("Level", "")
    scene_id = item.get("SceneID", "")
    clip     = item.get("ClipCode", "")
    if not (sub and ses and level and scene_id and clip):
        return ""
    scene_num = scene_id[len(level) + 1:]  # "w1l1s0" → "0"
    stem = f"sub-{sub}_ses-{ses}_task-mario_level-{level}_scene-{scene_num}_clip-{clip}"
    return str(gamelogs_reldir / f"{stem}{suffix}")


def _build_dataframe(summary_paths: list[Path], dataset_name: str) -> pd.DataFrame:
    rows = []
    relative_root = _common_root(summary_paths)
    for sp in summary_paths:
        try:
            with sp.open() as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Skipping %s: %s", sp, e)
            continue

        is_consolidated = isinstance(payload, list)
        items = payload if is_consolidated else [payload]
        sp_reldir = sp.parent.relative_to(relative_root) if relative_root else sp.parent

        for item in items:
            row = {col: item.get(col) for col in SUMMARY_COLUMNS}
            if is_consolidated:
                row["summary_path"] = _clip_relpath(item, sp_reldir)
            else:
                row["summary_path"] = str(sp.relative_to(relative_root)) if relative_root else str(sp)
            rows.append(row)

    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS + ["summary_path"])
    df["Cleared"] = (df["Outcome"] == "completed").astype("int8")
    df["dataset"] = dataset_name

    # Normalise BIDS entity prefixes — files use bare "01"/"001" strings.
    df["Subject"] = df["Subject"].astype(str).str.replace(r"^sub-", "", regex=True)
    df["Session"] = df["Session"].astype(str).str.replace(r"^ses-", "", regex=True)
    df["Run"] = df["Run"].astype(str).str.replace(r"^run-", "", regex=True)

    # ClipCode is the canonical ordinal time-axis; sort by it.
    df["ClipCode"] = df["ClipCode"].astype(str)
    df = df.sort_values(["Subject", "ClipCode"]).reset_index(drop=True)
    return df


def _common_root(paths: list[Path]) -> Path | None:
    """Longest common parent directory across the given paths, or None."""
    if not paths:
        return None
    parts = [list(p.resolve().parts) for p in paths]
    common: list[str] = []
    for chunk in zip(*parts, strict=False):
        if len(set(chunk)) == 1:
            common.append(chunk[0])
        else:
            break
    return Path(*common) if common else None


def _row_count(parquet_path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return pq.read_metadata(parquet_path).num_rows
    except Exception:
        return -1
