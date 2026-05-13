"""Invoke entry points for mario_learning.behavior.

Per-dataset tasks (e.g. `inv load`, `inv descriptive`) operate on a single
configured dataset; comparison tasks (`inv compare-*`) overlay multiple.

All tasks accept --dataset / --datasets selectors that resolve against
config.yaml > datasets, plus --subject / --session / --run filters, --force
to bypass cached outputs, and --slurm to dispatch on HPC.
"""

from __future__ import annotations

import logging
from pathlib import Path

from invoke import Collection, task

from mario_learning import provenance, utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mario_learning")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _selectors(subject: str | None, session: str | None, run: str | None):
    """Convert comma-separated CLI strings into sorted lists (or None)."""
    return (
        utils.split_csv(subject),
        utils.split_csv(session),
        utils.split_csv(run),
    )


def _resolve_dataset(cfg: dict, name: str) -> tuple[str, Path]:
    if name not in cfg["datasets"]:
        raise ValueError(
            f"Dataset '{name}' is not declared in config.yaml > datasets. "
            f"Available: {sorted(cfg['datasets'])}"
        )
    return name, Path(cfg["datasets"][name]["path"]).expanduser().resolve()


def _resolve_datasets(cfg: dict, names: str | None) -> list[tuple[str, Path]]:
    if names is None:
        names_list = list(cfg["datasets"].keys())
    else:
        names_list = [n.strip() for n in names.split(",") if n.strip()]
    return [_resolve_dataset(cfg, n) for n in names_list]


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def _load_clips_for_task(cfg, name, path, subject, session, run, force):
    """Shared loader call for every per-dataset analysis task."""
    from mario_learning import loader

    cache = utils.cache_dir(cfg, "load") / name
    subs, sess, runs = _selectors(subject, session, run)
    df = loader.load_clips(
        dataset_path=path,
        dataset_name=name,
        cache_dir=cache,
        subjects=subs,
        sessions=sess,
        runs=runs,
        force=force,
    )
    parameters = {
        "dataset_name": name,
        "subjects": subs,
        "sessions": sess,
        "runs": runs,
    }
    inputs = [cache / "clips.parquet"]
    return df, parameters, inputs


@task(help={
    "dataset": "Dataset name from config.yaml > datasets.",
    "subject": "Comma-separated subjects (e.g. 'sub-01,sub-02'). Default: all.",
    "session": "Comma-separated sessions. Default: all.",
    "run": "Comma-separated runs. Default: all.",
    "force": "Re-scan the dataset even if a cache exists.",
})
def load(c, dataset, subject=None, session=None, run=None, force=False):
    """Build or refresh the clips parquet cache for DATASET."""
    cfg = utils.load_config()
    name, path = _resolve_dataset(cfg, dataset)
    df, _, _ = _load_clips_for_task(cfg, name, path, subject, session, run, force)
    log.info("Loaded %d clips for dataset %s", len(df), name)


@task(help={
    "dataset": "Dataset name from config.yaml > datasets.",
    "subject": "Comma-separated subjects. Default: all.",
    "session": "Comma-separated sessions. Default: all.",
    "run": "Comma-separated runs. Default: all.",
    "force": "Recompute even if outputs and sidecars match.",
})
def descriptive(c, dataset, subject=None, session=None, run=None, force=False):
    """Per-subject / per-scene / per-phase / per-level descriptive tables + QC figure."""
    from mario_learning import descriptive as analysis

    cfg = utils.load_config()
    name, path = _resolve_dataset(cfg, dataset)
    df, parameters, inputs = _load_clips_for_task(cfg, name, path, subject, session, run, force)
    out_dir = utils.cache_dir(cfg, "descriptive") / name

    canary = out_dir / "subjects.csv"
    if not force and provenance.check_match(canary, parameters=parameters, inputs=inputs):
        log.info("descriptive %s: cache hit at %s", name, out_dir)
        return

    analysis.run(df, out_dir, parameters=parameters, inputs=inputs, cfg=cfg)
    log.info("descriptive %s -> %s", name, out_dir)


@task(name="learning-curves", help={
    "dataset": "Dataset name from config.yaml > datasets.",
    "subject": "Comma-separated subjects. Default: all.",
    "session": "Comma-separated sessions. Default: all.",
    "run": "Comma-separated runs. Default: all.",
    "force": "Recompute even if outputs and sidecars match.",
})
def learning_curves(c, dataset, subject=None, session=None, run=None, force=False):
    """Moving-average performance curves per subject/level/variable."""
    from mario_learning import learning_curves as analysis

    cfg = utils.load_config()
    name, path = _resolve_dataset(cfg, dataset)
    df, parameters, inputs = _load_clips_for_task(cfg, name, path, subject, session, run, force)
    parameters = {**parameters, "smoothing_window": cfg["analysis"]["learning_curves"]["smoothing_window"]}
    out_dir = utils.cache_dir(cfg, "learning_curves") / name

    canary = out_dir / "learning_curves.csv"
    if not force and provenance.check_match(canary, parameters=parameters, inputs=inputs):
        log.info("learning-curves %s: cache hit at %s", name, out_dir)
        return
    analysis.run(df, out_dir, parameters=parameters, inputs=inputs, cfg=cfg)
    log.info("learning-curves %s -> %s", name, out_dir)


@task(help={
    "dataset": "Dataset name from config.yaml > datasets.",
    "subject": "Comma-separated subjects. Default: all.",
    "session": "Comma-separated sessions. Default: all.",
    "run": "Comma-separated runs. Default: all.",
    "force": "Recompute even if outputs and sidecars match.",
})
def summary(c, dataset, subject=None, session=None, run=None, force=False):
    """Per-level grids of Cleared/Duration/Hits across scenes (with subject/phase overlays)."""
    from mario_learning import summary as analysis

    cfg = utils.load_config()
    name, path = _resolve_dataset(cfg, dataset)
    df, parameters, inputs = _load_clips_for_task(cfg, name, path, subject, session, run, force)
    out_dir = utils.cache_dir(cfg, "summary") / name

    canary = out_dir / "per_level_per_scene.csv"
    if not force and provenance.check_match(canary, parameters=parameters, inputs=inputs):
        log.info("summary %s: cache hit at %s", name, out_dir)
        return
    analysis.run(df, out_dir, parameters=parameters, inputs=inputs, cfg=cfg)
    log.info("summary %s -> %s", name, out_dir)


@task(name="scene-performance", help={
    "dataset": "Dataset name from config.yaml > datasets.",
    "subject": "Comma-separated subjects. Default: all.",
    "session": "Comma-separated sessions. Default: all.",
    "run": "Comma-separated runs. Default: all.",
    "force": "Recompute even if outputs and sidecars match.",
})
def scene_performance(c, dataset, subject=None, session=None, run=None, force=False):
    """Per-scene performance table + per (subject, level) facet-grid figure."""
    from mario_learning import scene_performance as analysis

    cfg = utils.load_config()
    name, path = _resolve_dataset(cfg, dataset)
    df, parameters, inputs = _load_clips_for_task(cfg, name, path, subject, session, run, force)
    out_dir = utils.cache_dir(cfg, "scene_performance") / name

    canary = out_dir / "per_scene.csv"
    if not force and provenance.check_match(canary, parameters=parameters, inputs=inputs):
        log.info("scene-performance %s: cache hit at %s", name, out_dir)
        return
    analysis.run(df, out_dir, parameters=parameters, inputs=inputs, cfg=cfg)
    log.info("scene-performance %s -> %s", name, out_dir)


@task(name="pattern-difficulty", help={
    "dataset": "Dataset name from config.yaml > datasets.",
    "subject": "Comma-separated subjects. Default: all.",
    "session": "Comma-separated sessions. Default: all.",
    "run": "Comma-separated runs. Default: all.",
    "force": "Recompute even if outputs and sidecars match.",
})
def pattern_difficulty(c, dataset, subject=None, session=None, run=None, force=False):
    """Clear rate by 27 scene patterns + learning metric. Downloads scene annotations on first run."""
    from mario_learning import pattern_difficulty as analysis

    cfg = utils.load_config()
    name, path = _resolve_dataset(cfg, dataset)
    df, parameters, inputs = _load_clips_for_task(cfg, name, path, subject, session, run, force)
    parameters = {**parameters, **cfg["analysis"]["pattern_difficulty"]}
    out_dir = utils.cache_dir(cfg, "pattern_difficulty") / name

    canary = out_dir / "pattern_metrics.csv"
    if not force and provenance.check_match(canary, parameters=parameters, inputs=inputs):
        log.info("pattern-difficulty %s: cache hit at %s", name, out_dir)
        return
    analysis.run(df, out_dir, parameters=parameters, inputs=inputs, cfg=cfg)
    log.info("pattern-difficulty %s -> %s", name, out_dir)


@task(help={
    "dataset": "Dataset name from config.yaml > datasets.",
    "subject": "Comma-separated subjects. Default: all.",
    "session": "Comma-separated sessions. Default: all.",
    "run": "Comma-separated runs. Default: all.",
    "force": "Recompute even if outputs and sidecars match.",
})
def clustering(c, dataset, subject=None, session=None, run=None, force=False):
    """UMAP scene-space + K-means clusters + per-cluster performance."""
    from mario_learning import clustering as analysis

    cfg = utils.load_config()
    name, path = _resolve_dataset(cfg, dataset)
    df, parameters, inputs = _load_clips_for_task(cfg, name, path, subject, session, run, force)
    parameters = {**parameters, **cfg["analysis"]["clustering"]}
    out_dir = utils.cache_dir(cfg, "clustering") / name

    canary = out_dir / "scenes_clustered.csv"
    if not force and provenance.check_match(canary, parameters=parameters, inputs=inputs):
        log.info("clustering %s: cache hit at %s", name, out_dir)
        return
    analysis.run(df, out_dir, parameters=parameters, inputs=inputs, cfg=cfg)
    log.info("clustering %s -> %s", name, out_dir)


@task(help={
    "dataset": "Dataset name from config.yaml > datasets.",
    "subject": "Comma-separated subjects. Default: all.",
    "session": "Comma-separated sessions. Default: all.",
    "run": "Comma-separated runs. Default: all.",
    "force": "Recompute even if outputs and sidecars match.",
})
def traces(c, dataset, subject=None, session=None, run=None, force=False):
    """Player x/y traces overlaid on level/scene background images. Needs _variables.json."""
    from mario_learning import traces as analysis

    cfg = utils.load_config()
    name, path = _resolve_dataset(cfg, dataset)
    df, parameters, inputs = _load_clips_for_task(cfg, name, path, subject, session, run, force)
    parameters = {**parameters, **cfg["analysis"]["traces"]}
    out_dir = utils.cache_dir(cfg, "traces") / name

    # Use a per-(subject, level) canary file as the idempotency probe.
    # Even one missing file per subject re-triggers the run; cheap enough.
    canaries = [out_dir / f"sub-{s}" / "levels" / f"{lvl}.png"
                for (s, lvl) in df[["Subject", "Level"]].drop_duplicates().itertuples(index=False)]
    if not force and canaries and all(provenance.check_match(p, parameters=parameters, inputs=inputs) for p in canaries):
        log.info("traces %s: cache hit at %s", name, out_dir)
        return
    analysis.run(df, out_dir, parameters=parameters, inputs=inputs, cfg=cfg, dataset_path=path)
    log.info("traces %s -> %s", name, out_dir)


@task(help={
    "dataset": "Dataset name from config.yaml > datasets.",
    "subject": "Comma-separated subjects. Default: all.",
    "session": "Comma-separated sessions. Default: all.",
    "run": "Comma-separated runs. Default: all.",
    "force": "Recompute even if outputs and sidecars match.",
})
def survival(c, dataset, subject=None, session=None, run=None, force=False):
    """Per-scene Kaplan-Meier survival curves (death or time-to-first-clear)."""
    from mario_learning import survival as analysis

    cfg = utils.load_config()
    name, path = _resolve_dataset(cfg, dataset)
    df, parameters, inputs = _load_clips_for_task(cfg, name, path, subject, session, run, force)
    parameters = {**parameters, **cfg["analysis"]["survival"]}
    out_dir = utils.cache_dir(cfg, "survival") / name

    canary = out_dir / "per_scene_summary.csv"
    if not force and provenance.check_match(canary, parameters=parameters, inputs=inputs):
        log.info("survival %s: cache hit at %s", name, out_dir)
        return
    analysis.run(df, out_dir, parameters=parameters, inputs=inputs, cfg=cfg)
    log.info("survival %s -> %s", name, out_dir)


@task(name="distribution-distances", help={
    "dataset": "Dataset name from config.yaml > datasets.",
    "subject": "Comma-separated subjects. Default: all.",
    "session": "Comma-separated sessions. Default: all.",
    "run": "Comma-separated runs. Default: all.",
    "force": "Recompute even if outputs and sidecars match.",
})
def distribution_distances(c, dataset, subject=None, session=None, run=None, force=False):
    """Wasserstein distances over scalar metrics + scene-space failure distributions."""
    from mario_learning import distribution_distances as analysis

    cfg = utils.load_config()
    name, path = _resolve_dataset(cfg, dataset)
    df, parameters, inputs = _load_clips_for_task(cfg, name, path, subject, session, run, force)
    parameters = {**parameters, **cfg["analysis"]["distribution_distances"]}
    out_dir = utils.cache_dir(cfg, "distribution_distances") / name

    canary = out_dir / "scalar.csv"
    if not force and provenance.check_match(canary, parameters=parameters, inputs=inputs):
        log.info("distribution-distances %s: cache hit at %s", name, out_dir)
        return

    # Prefer the dataset's own clustering UMAP cache if present; else None → Jaccard fallback.
    umap_coords = utils.cache_dir(cfg, "clustering") / name / "umap_2d.csv"
    umap_coords = umap_coords if umap_coords.exists() else None
    analysis.run(df, out_dir, parameters=parameters, inputs=inputs, cfg=cfg, umap_coords_path=umap_coords)
    log.info("distribution-distances %s -> %s", name, out_dir)


@task(help={
    "dataset": "Dataset name from config.yaml > datasets.",
    "subject": "Comma-separated subjects. Default: all.",
    "session": "Comma-separated sessions. Default: all.",
    "run": "Comma-separated runs. Default: all.",
    "force": "Recompute every step even if its sidecar matches.",
    "skip": "Comma-separated task names to skip (e.g. 'traces,clustering').",
})
def all(c, dataset, subject=None, session=None, run=None, force=False, skip=None):
    """Run every per-dataset analysis in order. Idempotent — skips steps whose sidecars match."""
    skip_set = set(utils.split_csv(skip) or [])
    sequence = [
        ("load", load),
        ("descriptive", descriptive),
        ("learning-curves", learning_curves),
        ("summary", summary),
        ("scene-performance", scene_performance),
        ("pattern-difficulty", pattern_difficulty),
        ("clustering", clustering),
        ("traces", traces),
        ("survival", survival),
        ("distribution-distances", distribution_distances),
    ]
    for step_name, fn in sequence:
        if step_name in skip_set:
            log.info("all: skipping %s (requested)", step_name)
            continue
        log.info("all: running %s ...", step_name)
        kwargs = {"dataset": dataset, "subject": subject, "session": session, "run": run, "force": force}
        if step_name == "load":
            kwargs.pop("force", None)
            kwargs["force"] = force
        fn(c, **kwargs)


# ---------------------------------------------------------------------------
# Cross-dataset comparison tasks
# ---------------------------------------------------------------------------

def _resolve_compare_inputs(cfg, datasets_csv: str | None, task_name: str) -> tuple[dict[str, Path], list[Path]]:
    """Return {dataset_name: per_dataset_cache_dir} for the requested datasets."""
    pairs = _resolve_datasets(cfg, datasets_csv)
    out = {}
    inputs = []
    for name, _ in pairs:
        d = utils.cache_dir(cfg, task_name) / name
        out[name] = d
        inputs.append(d)
    return out, inputs


def _run_compare(c, task_name: str, datasets_csv, force, runner):
    cfg = utils.load_config()
    sources, inputs = _resolve_compare_inputs(cfg, datasets_csv, task_name)
    out_dir = utils.cache_dir(cfg, "compare") / task_name
    parameters = {"datasets": sorted(sources)}
    canary = out_dir / "_compare.json"
    if not force and provenance.check_match(canary, parameters=parameters, inputs=inputs):
        log.info("compare-%s: cache hit at %s", task_name, out_dir)
        return
    runner(sources, out_dir, parameters=parameters, inputs=inputs, cfg=cfg)
    canary.write_text("ok\n")
    provenance.write_sidecar(canary, parameters=parameters, inputs=inputs)


@task(name="compare-descriptive", help={"datasets": "Comma-separated dataset names. Default: all configured.", "force": "Recompute."})
def compare_descriptive(c, datasets=None, force=False):
    """Overlay descriptive tables across datasets."""
    from mario_learning import compare as cmp
    _run_compare(c, "descriptive", datasets, force, cmp.descriptive)


@task(name="compare-learning-curves", help={"datasets": "Comma-separated dataset names.", "force": "Recompute."})
def compare_learning_curves(c, datasets=None, force=False):
    """Overlay smoothed learning curves across datasets."""
    from mario_learning import compare as cmp
    _run_compare(c, "learning_curves", datasets, force, cmp.learning_curves)


@task(name="compare-scene-performance", help={"datasets": "Comma-separated dataset names.", "force": "Recompute."})
def compare_scene_performance(c, datasets=None, force=False):
    """Compare per-scene performance across datasets."""
    from mario_learning import compare as cmp
    _run_compare(c, "scene_performance", datasets, force, cmp.scene_performance)


@task(name="compare-pattern-difficulty", help={"datasets": "Comma-separated dataset names.", "force": "Recompute."})
def compare_pattern_difficulty(c, datasets=None, force=False):
    """Compare pattern-difficulty metrics across datasets."""
    from mario_learning import compare as cmp
    _run_compare(c, "pattern_difficulty", datasets, force, cmp.pattern_difficulty)


@task(name="compare-clustering", help={"datasets": "Comma-separated dataset names.", "force": "Recompute."})
def compare_clustering(c, datasets=None, force=False):
    """Overlay UMAP scene embeddings across datasets."""
    from mario_learning import compare as cmp
    _run_compare(c, "clustering", datasets, force, cmp.clustering)


@task(name="compare-summary", help={"datasets": "Comma-separated dataset names.", "force": "Recompute."})
def compare_summary(c, datasets=None, force=False):
    """Concat per-level/per-scene summary tables across datasets."""
    from mario_learning import compare as cmp
    _run_compare(c, "summary", datasets, force, cmp.summary)


@task(name="compare-traces", help={"datasets": "Comma-separated dataset names.", "force": "Recompute."})
def compare_traces(c, datasets=None, force=False):
    """Build a coverage table across each dataset's trace PNGs."""
    from mario_learning import compare as cmp
    _run_compare(c, "traces", datasets, force, cmp.traces)


@task(name="compare-survival", help={"datasets": "Comma-separated dataset names.", "force": "Recompute."})
def compare_survival(c, datasets=None, force=False):
    """Overlay per-scene KM curves across datasets."""
    from mario_learning import compare as cmp
    _run_compare(c, "survival", datasets, force, cmp.survival)


@task(name="compare-distribution-distances", help={"datasets": "Comma-separated dataset names.", "force": "Recompute."})
def compare_distribution_distances(c, datasets=None, force=False):
    """Pairwise scalar + scene-space EMD between datasets, with heatmap."""
    from mario_learning import compare as cmp
    _run_compare(c, "distribution_distances", datasets, force, cmp.distribution_distances)


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------

ns = Collection()
ns.add_task(load)
ns.add_task(descriptive)
ns.add_task(learning_curves)
ns.add_task(summary)
ns.add_task(scene_performance)
ns.add_task(pattern_difficulty)
ns.add_task(clustering)
ns.add_task(traces)
ns.add_task(survival)
ns.add_task(distribution_distances)
ns.add_task(all)
ns.add_task(compare_descriptive)
ns.add_task(compare_learning_curves)
ns.add_task(compare_scene_performance)
ns.add_task(compare_pattern_difficulty)
ns.add_task(compare_clustering)
ns.add_task(compare_summary)
ns.add_task(compare_traces)
ns.add_task(compare_survival)
ns.add_task(compare_distribution_distances)
