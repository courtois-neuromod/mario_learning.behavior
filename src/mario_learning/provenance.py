"""Provenance sidecars per .AGENTS.md.

Every generated artifact (parquet/csv/png/...) gets a sibling JSON sidecar with
script identity, git commit, parameters, input fingerprints, and seed. Tasks
use `check_match()` for idempotency: re-running with matching parameters and
unchanged inputs is a no-op unless `--force` is passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from mario_learning.utils import REPO_ROOT, git_commit


def _fingerprint_path(p: Path) -> dict[str, Any]:
    """File fingerprint: relative path (when under the repo), size, mtime."""
    p = Path(p)
    try:
        st = p.stat()
        size, mtime = st.st_size, st.st_mtime
    except FileNotFoundError:
        size, mtime = None, None
    try:
        rel = str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        rel = str(p)
    return {"path": rel, "size": size, "mtime": mtime}


def fingerprint_inputs(inputs: list[str | os.PathLike]) -> str:
    """Hash a list of input paths into a stable short digest."""
    payload = [_fingerprint_path(Path(p)) for p in sorted(map(str, inputs))]
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def sidecar_path(output_path: str | os.PathLike) -> Path:
    """`<output>.json` — same base path, `.json` appended."""
    p = Path(output_path)
    return p.with_suffix(p.suffix + ".json")


def write_sidecar(
    output_path: str | os.PathLike,
    *,
    parameters: dict[str, Any],
    inputs: list[str | os.PathLike],
    random_seed: int | None = None,
    script: str | None = None,
) -> Path:
    """Write a JSON sidecar next to `output_path`."""
    payload = {
        "script": script or _calling_script(),
        "git_commit": git_commit(),
        "date": datetime.now().isoformat(timespec="seconds"),
        "parameters": parameters,
        "inputs": [_fingerprint_path(Path(p)) for p in inputs],
        "inputs_digest": fingerprint_inputs(inputs),
        "random_seed": random_seed,
    }
    side = sidecar_path(output_path)
    side.parent.mkdir(parents=True, exist_ok=True)
    with side.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return side


def check_match(
    output_path: str | os.PathLike,
    *,
    parameters: dict[str, Any],
    inputs: list[str | os.PathLike],
) -> bool:
    """Return True if the existing output and sidecar both match the request."""
    out = Path(output_path)
    side = sidecar_path(out)
    if not out.exists() or not side.exists():
        return False
    try:
        with side.open() as f:
            existing = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if existing.get("parameters") != parameters:
        return False
    if existing.get("inputs_digest") != fingerprint_inputs(inputs):
        return False
    return True


def _calling_script() -> str:
    """Best-effort caller identifier, relative to the repo root."""
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    if main:
        try:
            return str(Path(main).resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(Path(main))
    return "<unknown>"
