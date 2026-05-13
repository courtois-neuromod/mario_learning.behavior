"""Loader smoke tests on the fixture dataset."""

from __future__ import annotations

from mario_learning import loader


def test_loads_all_clips(fixture_dataset, cache_dir):
    df = loader.load_clips(
        dataset_path=fixture_dataset,
        dataset_name="humans_tiny",
        cache_dir=cache_dir,
    )
    # 2 subjects × 2 sessions × 2 runs × 4 scenes × 3 reps = 96 clips
    assert len(df) == 96
    assert set(df["Subject"]) == {"01", "02"}
    assert set(df["Session"]) == {"001", "002"}
    assert set(df["Run"]) == {"01", "02"}
    assert set(df["Phase"]) == {"discovery", "practice"}
    assert "Cleared" in df.columns
    assert df["Cleared"].between(0, 1).all()


def test_subject_filter(fixture_dataset, cache_dir):
    df = loader.load_clips(
        dataset_path=fixture_dataset,
        dataset_name="humans_tiny",
        cache_dir=cache_dir,
        subjects=["sub-01"],
    )
    assert set(df["Subject"]) == {"01"}
    assert len(df) == 48  # half


def test_run_filter_post_dataframe(fixture_dataset, cache_dir):
    """Run filter is applied after dataframe build (run lives in JSON payload)."""
    df = loader.load_clips(
        dataset_path=fixture_dataset,
        dataset_name="humans_tiny",
        cache_dir=cache_dir,
        runs=["run-01"],
    )
    assert set(df["Run"]) == {"01"}


def test_cache_hits_on_second_call(fixture_dataset, cache_dir):
    first = loader.load_clips(
        dataset_path=fixture_dataset,
        dataset_name="humans_tiny",
        cache_dir=cache_dir,
    )
    second = loader.load_clips(
        dataset_path=fixture_dataset,
        dataset_name="humans_tiny",
        cache_dir=cache_dir,
    )
    assert (first == second).all().all()
    assert (cache_dir / "clips.parquet").exists()
    assert (cache_dir / "clips.parquet.json").exists()
