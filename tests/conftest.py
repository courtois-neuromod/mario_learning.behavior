"""Build a tiny mario.scenes-formatted fixture dataset on disk.

Three subjects × two sessions × two runs, with handcrafted clip summaries that
exercise the loader, descriptive stats, learning curves, and survival logic
without depending on the real ~50GB mario.scenes data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _summary(
    subject: str,
    session: str,
    run: str,
    level: str,
    scene_idx: int,
    rep: int,
    start_frame: int,
    duration_frames: int,
    outcome: str,
    phase: str,
) -> dict:
    end_frame = start_frame + duration_frames
    clip_code = f"{int(session):03d}{int(run):02d}{rep:02d}{start_frame:07d}"
    return {
        "Subject": subject,
        "Session": session,
        "Run": run,
        "Level": level,
        "SceneID": f"{level}s{scene_idx}",
        "ClipCode": clip_code,
        "StartFrame": start_frame,
        "EndFrame": end_frame,
        "Duration": round(duration_frames / 60.0, 3),
        "Outcome": outcome,
        "Phase": phase,
        "SourceBk2": f"sub-{subject}/ses-{session}/gamelogs/source.bk2",
        "GameName": "SuperMarioBros-Nes",
        "ScoreGained": 100 * rep,
        "CoinsGained": rep,
        "Lives_lost": 0 if outcome == "completed" else 1,
        "Hits_taken": 0 if outcome == "completed" else 1,
        "Enemies_killed": rep,
        "Powerups_collected": 0,
        "Bricks_smashed": 0,
        "X_Traveled": 150 + rep * 10,
    }


def _write_clip(gamelogs: Path, summary: dict, *, frames: int = 60) -> None:
    base = (
        f"sub-{summary['Subject']}_ses-{summary['Session']}"
        f"_task-mario_level-{summary['Level']}"
        f"_scene-{summary['SceneID'][-1]}_clip-{summary['ClipCode']}"
    )
    with (gamelogs / f"{base}_summary.json").open("w") as f:
        json.dump(summary, f)
    # Minimal variables.json: only the keys the traces task expects.
    variables = {
        "player_x_posHi": [0] * frames,
        "player_x_posLo": list(range(100, 100 + frames)),
        "player_y_pos": [176] * frames,
    }
    with (gamelogs / f"{base}_variables.json").open("w") as f:
        json.dump(variables, f)


@pytest.fixture(scope="session")
def fixture_dataset(tmp_path_factory) -> Path:
    """Materialize ``humans_tiny`` — 2 subjects × 2 sessions × 2 runs × 4 scenes."""
    root = tmp_path_factory.mktemp("humans_tiny")
    rng_seed = 0
    for subject in ("01", "02"):
        for session in ("001", "002"):
            for run in ("01", "02"):
                level = "w1l1" if run == "01" else "w1l2"
                phase = "discovery" if session == "001" else "practice"
                gamelogs = root / f"sub-{subject}" / f"ses-{session}" / "gamelogs"
                gamelogs.mkdir(parents=True, exist_ok=True)
                for scene_idx in range(4):
                    for rep in range(3):
                        # Subject 01 in practice clears more often; sub 02 fails more.
                        if subject == "02" and rep == 0 and scene_idx % 2 == 0:
                            outcome = "death"
                        else:
                            outcome = "completed" if (rng_seed + scene_idx + rep) % 5 else "death"
                        rng_seed += 1
                        start = 100 + scene_idx * 50 + rep
                        summary = _summary(subject, session, run, level, scene_idx, rep,
                                           start_frame=start, duration_frames=60 + rep * 5,
                                           outcome=outcome, phase=phase)
                        _write_clip(gamelogs, summary)
    return root


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    return tmp_path / "cache"
