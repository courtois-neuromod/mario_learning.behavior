"""Shared helpers: config loading, BIDS entity parsing, paths, git provenance.

Also contains the scene-annotation loader (`load_scene_annotations`) — a port
of mario.scenes' `ensure_scenes_data` + `load_scenes_info` that downloads the
27-feature CSV from Zenodo on first call and caches it under the repo root.
"""

from __future__ import annotations

import json
import logging
import os
import os.path as op
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

BIDS_ENTITY_RE = re.compile(r"(?P<key>[a-zA-Z]+)-(?P<value>[A-Za-z0-9]+)")


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load `config.yaml` from the repo root (or a custom path)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found. Run ./setup.sh or copy config.yaml.template."
        )
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def output_dir(cfg: dict, task_name: str) -> Path:
    """`<repo>/<paths.output>/<task_name>/` — creates the parent on demand."""
    base = REPO_ROOT / cfg["paths"]["output"] / task_name
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_dir(cfg: dict, task_name: str) -> Path:
    """Deprecated alias for `output_dir`. Kept for backwards compatibility."""
    return output_dir(cfg, task_name)


def logs_dir(cfg: dict) -> Path:
    base = REPO_ROOT / cfg["paths"]["logs"]
    base.mkdir(parents=True, exist_ok=True)
    return base


def split_csv(value: str | None) -> list[str] | None:
    """'a,b,c' → ['a','b','c']; None → None; '' → None."""
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return parts or None


def parse_bids_entities(path: str | os.PathLike) -> dict[str, str]:
    """Extract BIDS entities (`sub`, `ses`, `run`, `task`, ...) from a filename.

    Example: ``sub-01_ses-002_task-mario_run-03_..._summary.json`` →
    ``{'sub': '01', 'ses': '002', 'task': 'mario', 'run': '03'}``.
    """
    name = op.basename(str(path))
    return {m.group("key"): m.group("value") for m in BIDS_ENTITY_RE.finditer(name)}


def git_commit() -> str | None:
    """Return the current git commit short hash, or None if outside a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def dataset_name_from_path(path: str | os.PathLike) -> str:
    """Fallback dataset name when none is configured: use the directory name."""
    return Path(path).expanduser().resolve().name


SCENE_FEATURE_COLS = [
    "Enemy", "2-Horde", "3-Horde", "4-Horde", "Roof", "Gap",
    "Multiple gaps", "Variable gaps", "Gap enemy", "Pillar gap", "Valley",
    "Pipe valley", "Empty valley", "Enemy valley", "Roof valley", "2-Path",
    "3-Path", "Risk/Reward", "Stair up", "Stair down", "Empty stair valley",
    "Enemy stair valley", "Gap stair valley", "Reward", "Moving platform",
    "Flagpole", "Beginning", "Bonus zone",
]

ZENODO_RECORD = "15586709"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def _scenes_cache_path() -> Path:
    return REPO_ROOT / "data" / "external" / "scenes_mastersheet.csv"


def ensure_scenes_mastersheet() -> Path:
    """Download `scenes_mastersheet.csv` from Zenodo if it's not already cached."""
    target = _scenes_cache_path()
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Fetching Zenodo record %s metadata ...", ZENODO_RECORD)
    with urllib.request.urlopen(ZENODO_API_URL) as resp:
        record = json.loads(resp.read().decode())
    latest_url = record.get("links", {}).get("latest")
    if latest_url:
        with urllib.request.urlopen(latest_url) as resp:
            record = json.loads(resp.read().decode())
    csv_url = None
    for f in record.get("files", []):
        if f["key"].endswith("scenes_mastersheet.csv"):
            csv_url = f["links"]["self"]
            break
    if csv_url is None:
        raise FileNotFoundError(
            f"scenes_mastersheet.csv not found in Zenodo record {ZENODO_RECORD}."
        )
    logger.info("Downloading %s -> %s", csv_url, target)
    urllib.request.urlretrieve(csv_url, target)
    return target


def load_scene_annotations() -> pd.DataFrame:
    """Return a DataFrame of scene annotations indexed by ``scene_id``."""
    df = pd.read_csv(ensure_scenes_mastersheet()).dropna(subset=["World", "Level", "Scene"])
    df["scene_id"] = df.apply(
        lambda r: f"w{int(r['World'])}l{int(r['Level'])}s{int(r['Scene'])}",
        axis=1,
    )
    return df


def scene_pattern_matrix() -> pd.DataFrame:
    """Binary feature matrix (n_scenes × 28), indexed by ``scene_id``."""
    df = load_scene_annotations()
    out = df[SCENE_FEATURE_COLS].astype(int).copy()
    out.index = df["scene_id"].values
    out.index.name = "scene_id"
    return out


def list_summary_jsons(
    dataset_path: Path,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
) -> list[Path]:
    """Enumerate ``*_summary.json`` paths under a mario.scenes-formatted root.

    Finds both per-clip files (``sub-*/ses-*/gamelogs/*_summary.json``) and
    consolidated per-session files (``sub-*/ses-*/*_summary.json``).
    Filters by ``sub`` and ``ses`` BIDS entities parsed from each filename.
    Run-level filtering is applied at the DataFrame level by the loader because
    ``*_summary.json`` filenames carry ``level`` and ``scene`` entities
    rather than ``run`` — the run number lives only inside the JSON payload.
    """
    files: list[Path] = []
    all_summaries = sorted(
        set(dataset_path.glob("sub-*/ses-*/gamelogs/*_summary.json"))
        | set(dataset_path.glob("sub-*/ses-*/*_summary.json"))
    )
    for summary in all_summaries:
        ents = parse_bids_entities(summary)
        if subjects and f"sub-{ents.get('sub')}" not in subjects:
            continue
        if sessions and f"ses-{ents.get('ses')}" not in sessions:
            continue
        files.append(summary)
    return files


def strip_bids_prefix(values: list[str] | None, prefix: str) -> list[str] | None:
    """`['sub-01']` → `['01']`. Returns None if values is None/empty."""
    if not values:
        return None
    return [v[len(prefix):] if v.startswith(prefix) else v for v in values]


STAGES = ["early_discovery", "late_discovery", "early_practice", "late_practice"]


def add_stage_column(clips: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `clips` with a `Stage` column ∈ {early|late}_{discovery|practice}.

    Within each (Subject, Phase, Level) the split is at the per-group median
    ClipCode using strict ``<`` (the median itself goes to *late*). The split
    is per-level so that every (pattern, phase) cell sees both early and late
    representation — without it, a level played entirely at the start of
    discovery would put all its clips in early_discovery, leaving late_discovery
    empty for the patterns it contains (e.g. 4-Horde, Enemy stair valley).
    """
    out = clips.copy()
    out["Stage"] = pd.Series(dtype="object", index=out.index)
    codes = clips["ClipCode"].astype("int64")
    median = codes.groupby([clips["Subject"], clips["Phase"], clips["Level"]]).transform("median")
    early = codes < median
    out["Stage"] = np.where(early, "early_", "late_") + clips["Phase"].astype(str)
    return out


def attach_patterns(clips: pd.DataFrame) -> pd.DataFrame:
    """Long-form (one row per clip × pattern flag = 1) join of clips with scene patterns.

    The 27 scene-pattern columns from the Zenodo annotations are melted, only
    rows where the pattern is present are kept, and the result is joined to
    `clips` on `SceneID`. Use this whenever a per-pattern aggregation is
    needed (per [[feedback-patterns-over-scenes]]).
    """
    patterns = scene_pattern_matrix().reset_index().melt(
        id_vars="scene_id", var_name="pattern", value_name="present"
    )
    patterns = patterns[patterns["present"] == 1].drop(columns="present")
    return clips.merge(patterns, left_on="SceneID", right_on="scene_id", how="inner").drop(columns="scene_id")
