"""End-to-end smoke tests for each analysis on the fixture dataset."""

from __future__ import annotations

import pandas as pd

from mario_learning import (
    descriptive,
    distribution_distances,
    learning_curves,
    loader,
    scene_performance,
    summary,
    survival,
)

CFG = {
    "analysis": {
        "learning_curves": {"smoothing_window": 3, "variables": ["Cleared", "Hits_taken"]},
        "survival": {"event": "death", "censor_at": "clip_end", "min_attempts_per_scene": 3},
        "distribution_distances": {
            "scalar_variables": ["Duration", "Hits_taken"],
            "scene_space": {"ground_metric": "jaccard"},
            "grouping": "phase",
        },
        "figure": {"dpi": 72, "palette": "tab10"},
    },
}


def _clips(fixture_dataset, cache_dir):
    return loader.load_clips(
        dataset_path=fixture_dataset,
        dataset_name="humans_tiny",
        cache_dir=cache_dir,
    )


def test_descriptive(fixture_dataset, cache_dir, tmp_path):
    clips = _clips(fixture_dataset, cache_dir)
    paths = descriptive.run(
        clips, tmp_path / "out",
        parameters={}, inputs=[], cfg=CFG,
    )
    assert {"subjects", "scenes", "phases", "levels", "qc"} <= set(paths)
    for p in paths.values():
        assert p.exists()
    subjects = pd.read_csv(paths["subjects"])
    assert len(subjects) == 2


def test_learning_curves(fixture_dataset, cache_dir, tmp_path):
    clips = _clips(fixture_dataset, cache_dir)
    paths = learning_curves.run(
        clips, tmp_path / "out",
        parameters={}, inputs=[], cfg=CFG,
    )
    assert paths["table"].exists()
    smoothed = pd.read_csv(paths["table"])
    assert "Cleared_smoothed" in smoothed.columns


def test_summary(fixture_dataset, cache_dir, tmp_path):
    clips = _clips(fixture_dataset, cache_dir)
    paths = summary.run(clips, tmp_path / "out", parameters={}, inputs=[], cfg=CFG)
    assert paths["table"].exists()
    # 2 levels in the fixture
    assert sum(1 for k in paths if k.startswith("figure_")) == 2


def test_scene_performance(fixture_dataset, cache_dir, tmp_path):
    clips = _clips(fixture_dataset, cache_dir)
    paths = scene_performance.run(clips, tmp_path / "out", parameters={}, inputs=[], cfg=CFG)
    assert paths["table"].exists()


def test_survival(fixture_dataset, cache_dir, tmp_path):
    clips = _clips(fixture_dataset, cache_dir)
    paths = survival.run(clips, tmp_path / "out", parameters={}, inputs=[], cfg=CFG)
    summary_df = pd.read_csv(paths["summary"])
    assert (summary_df["n_observations"] >= CFG["analysis"]["survival"]["min_attempts_per_scene"]).all()


def test_distribution_distances(fixture_dataset, cache_dir, tmp_path):
    clips = _clips(fixture_dataset, cache_dir)
    paths = distribution_distances.run(
        clips, tmp_path / "out",
        parameters={}, inputs=[], cfg=CFG, umap_coords_path=None,
    )
    scalar = pd.read_csv(paths["scalar"])
    # phase grouping → 1 pair (discovery vs practice) × 2 variables
    assert len(scalar) == 2
