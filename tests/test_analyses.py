"""End-to-end smoke tests for each analysis on the fixture dataset."""

from __future__ import annotations

import pandas as pd

from mario_learning import (
    descriptive,
    distribution_distances,
    learning_curves,
    loader,
    pattern_difficulty,
    pattern_performance,
    summary,
    survival,
    utils,
)

CFG = {
    "analysis": {
        "learning_curves": {"smoothing_window": 3, "variables": ["Cleared", "Hits_taken"]},
        "pattern_difficulty": {"phases": ["discovery", "practice"], "early_late_split": "median"},
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


def test_stage_helper(fixture_dataset, cache_dir):
    clips = _clips(fixture_dataset, cache_dir)
    staged = utils.add_stage_column(clips)
    assert set(staged["Stage"]) <= set(utils.STAGES)
    # Every (subject, phase) should only contain valid stages for that phase.
    for (_subject, phase), g in staged.groupby(["Subject", "Phase"]):
        stages = set(g["Stage"])
        valid = {f"early_{phase}", f"middle_{phase}", f"late_{phase}"}
        assert stages <= valid


def test_attach_patterns(fixture_dataset, cache_dir):
    clips = _clips(fixture_dataset, cache_dir)
    long_df = utils.attach_patterns(clips)
    assert "pattern" in long_df.columns
    # The fixture uses real scene IDs that should match at least one pattern each.
    assert len(long_df) > len(clips)


def test_descriptive(fixture_dataset, cache_dir, tmp_path):
    clips = _clips(fixture_dataset, cache_dir)
    paths = descriptive.run(
        clips, tmp_path / "out",
        parameters={}, inputs=[], cfg=CFG,
    )
    assert {"subjects", "subjects_stages", "subjects_patterns", "scenes", "phases", "levels", "qc"} <= set(paths)
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
    for key in ("table_patterns", "table_levels"):
        assert paths[key].exists()
    smoothed = pd.read_csv(paths["table_patterns"])
    assert "Cleared_smoothed" in smoothed.columns
    assert "pattern" in smoothed.columns


def test_summary(fixture_dataset, cache_dir, tmp_path):
    clips = _clips(fixture_dataset, cache_dir)
    paths = summary.run(clips, tmp_path / "out", parameters={}, inputs=[], cfg=CFG)
    assert paths["table"].exists()
    # One figure per subject (2 in the fixture).
    assert sum(1 for k in paths if k.startswith("figure_")) == 2


def test_pattern_performance(fixture_dataset, cache_dir, tmp_path):
    clips = _clips(fixture_dataset, cache_dir)
    paths = pattern_performance.run(clips, tmp_path / "out", parameters={}, inputs=[], cfg=CFG)
    table = pd.read_csv(paths["table"])
    assert {"Subject", "pattern", "Cleared"} <= set(table.columns)


def test_pattern_difficulty_four_stages(fixture_dataset, cache_dir, tmp_path):
    clips = _clips(fixture_dataset, cache_dir)
    paths = pattern_difficulty.run(clips, tmp_path / "out", parameters={}, inputs=[], cfg=CFG)
    metrics = pd.read_csv(paths["table"])
    # All 4 stage columns present + the improvement column.
    for col in (*utils.STAGES, "improvement"):
        assert col in metrics.columns


def test_survival_per_pattern(fixture_dataset, cache_dir, tmp_path):
    clips = _clips(fixture_dataset, cache_dir)
    paths = survival.run(clips, tmp_path / "out", parameters={}, inputs=[], cfg=CFG)
    summary_df = pd.read_csv(paths["pattern_summary"])
    assert (summary_df["n_observations"] >= CFG["analysis"]["survival"]["min_attempts_per_scene"]).all()
    assert {"Subject", "pattern"} <= set(summary_df.columns)


def test_distribution_distances(fixture_dataset, cache_dir, tmp_path):
    clips = _clips(fixture_dataset, cache_dir)
    paths = distribution_distances.run(
        clips, tmp_path / "out",
        parameters={}, inputs=[], cfg=CFG, umap_coords_path=None,
    )
    scalar = pd.read_csv(paths["scalar"])
    # phase grouping → 1 pair (discovery vs practice) × 2 variables
    assert len(scalar) == 2


def test_distribution_distances_subject_stage(fixture_dataset, cache_dir, tmp_path):
    clips = _clips(fixture_dataset, cache_dir)
    cfg = {**CFG, "analysis": {**CFG["analysis"]}}
    cfg["analysis"]["distribution_distances"] = {
        **CFG["analysis"]["distribution_distances"],
        "grouping": "subject_stage",
    }
    paths = distribution_distances.run(
        clips, tmp_path / "out",
        parameters={}, inputs=[], cfg=cfg, umap_coords_path=None,
    )
    scalar = pd.read_csv(paths["scalar"])
    # Within each subject (2 in fixture), 6 stages → C(6,2)=15 pairs × 2 vars = 30 rows per subject, 60 total.
    assert len(scalar) == 60
    # No cross-subject pairs.
    for _, row in scalar.iterrows():
        a_sub = row["group_a"].split("_", 1)[0]
        b_sub = row["group_b"].split("_", 1)[0]
        assert a_sub == b_sub
